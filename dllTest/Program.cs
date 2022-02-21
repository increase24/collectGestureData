using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace gestureCollect
{
    class Program
    {
        [DllImport("handLandmarks.dll", EntryPoint = "handLandmarks_Init", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.StdCall)]
        public static extern IntPtr handLandmarks_Init(string p_palmDetModel, string p_handKptsDetModel, string p_handGesRecModel, string p_anchorFile, int[] image_shape, string root_path);
        [DllImport("handLandmarks.dll", EntryPoint = "handLandmarks_inference", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.StdCall)]
        public static extern int handLandmarks_inference(IntPtr p_self, IntPtr image, int[] image_shape, float[] output, int label_gesture, string imagePrefix, bool flag_record);
        static void Main(string[] args)
        {
            Console.WriteLine("Start collecting data, Let's go!");
            string pamlDetModelPath = "./models/palm_detection.onnx"; // "D:/software/projects/c/hand/collectHandData/build/Release/models/palm_detection.onnx";
            string handKptsDetModelPath = "./models/blaze_hand.onnx";  //"D:/software/projects/c/hand/collectHandData/build/Release/models/blaze_hand.onnx";
            string handGesRecModelPath = "./models/mobilenetV2_hw32_c10.onnx";
            string anchorFilePath = "./models/anchors.bin";  //"D:/software/projects/c/hand/collectHandData/build/Release/models/anchors.bin";
            int[] imgShape_model = { 256, 256 };
            string[] gesture_classes = { "number_1", "number_2", "number_3", "another_number_3", "number_4", "number_5", "number_6", "thumb_up", "ok", "heart"};
            string curr_time = System.DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss");
            //make_dir(curr_time, gesture_classes);
            IntPtr p_session = handLandmarks_Init(pamlDetModelPath, handKptsDetModelPath, handGesRecModelPath, anchorFilePath, imgShape_model, curr_time);
            Console.WriteLine("Successfully initialize the network");
            VideoCapture cap = new VideoCapture();
            Console.Write("Please enter the number of collection cycles:");
            int epoches;
            try
            {
                string collection_cycles = Console.ReadLine();
                epoches = int.Parse(collection_cycles);
            }
            catch
            {
                epoches = 1;
            }
            Console.Write("Please input the camera index:");
            string camera_index = Console.ReadLine();
            cap.Open(int.Parse(camera_index));
            int rawWidth = cap.FrameWidth;
            int rawHeight = cap.FrameHeight;
            double fps = cap.Fps;
            Console.WriteLine("FPS: " + fps.ToString());
            Mat rawFrame = new Mat();
            int[] image_shape = new int[] {rawHeight, rawWidth };
            float[] output_arr = new float[89];
            bool flag_first = true;
            int flag_quit;
            int epoch = 0, i_class=0, counter = 0;
            int num_sec_per_epoch = 10;
            int num_samples_per_epoch = (int)(num_sec_per_epoch * fps);
            Console.WriteLine("number of samples per class in one epoch: " + num_samples_per_epoch.ToString());
            while (true)
            {
                bool hasFrame = cap.Read(rawFrame);

                if (!hasFrame)
                {
                    cap.Release();
                    Cv2.DestroyAllWindows();
                    break;
                }
                else
                {
                    IntPtr p_input = rawFrame.Data;
                    if (counter < (int)(2 * fps))
                    {
                        flag_quit = handLandmarks_inference(p_session, p_input, image_shape, output_arr, i_class, epoch.ToString() + "_" + counter.ToString(), false);
                    }
                    else
                    {
                        flag_quit = handLandmarks_inference(p_session, p_input, image_shape, output_arr, i_class, epoch.ToString() + "_" + counter.ToString(), true);
                    }
                    
                    counter++;
                    if (counter == num_samples_per_epoch)
                    {
                        counter = 0;
                        i_class++;
                        if (i_class == gesture_classes.Length)
                        {
                            i_class = 0;
                            epoch++;
                            if (epoch == epoches)
                            {
                                cap.Release();
                                //Cv2.DestroyAllWindows();
                                break;
                            }
                        }
                    }
                    if (flag_quit == 1)
                    {
                        cap.Release();
                        //Cv2.DestroyAllWindows();
                        break;
                    }
                    //if (flag_first)
                    //{
                    //    handLandmarks_inference(p_session, p_input, image_shape, output_arr, true, true);
                    //    flag_first = false;
                    //}
                    //else
                    //{
                    //    handLandmarks_inference(p_session, p_input, image_shape, output_arr, true, true);
                    //}

                }
                //Cv2.ImShow("LandmarkDetection", rawFrame);
            }
        }

        static void make_dir(string rootPath, string[] classes)
        {
            if (false == System.IO.Directory.Exists(rootPath))
            {
                System.IO.Directory.CreateDirectory(rootPath);
            }
            string[] img_type = {"/image_fullRegion/", "/image_handRegion/"};
            foreach (string _type in img_type)
            {
                if (false == System.IO.Directory.Exists(rootPath + _type))
                {
                    System.IO.Directory.CreateDirectory(rootPath + _type);
                }
                foreach (string _class in classes)
                {
                    if (false == System.IO.Directory.Exists(rootPath + _type + _class))
                    {
                        System.IO.Directory.CreateDirectory(rootPath + _type + _class);
                    }
                }
            }
        }
    }
}
