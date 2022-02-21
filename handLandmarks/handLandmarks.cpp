#include "pch.h"
#define _DLL_EXPORTS
#include "handLandmarks.h"

//int modelWidth = 256, modelHeight = 256;
//int numAnchors = 2944, outDim = 18, batchSizePalm = 1, batchSizeHand = 1;
//int numKeypointsPalm = 7, numKeypointsHand = 21, numJointConnect = 20, palmDetFreq = 20;
//float scoreClipThrs = 100.0, minScoreThrs = 0.80, minSuppressionThrs = 0.3, handThrs = 0.8;
//float palm_shift_y = 0.5, palm_shift_x = 0, palm_box_scale = 2.6,
//hand_shift_y = 0, hand_shift_x = 0, hand_box_scale = 2.1;

cv::Mat load_gestureImg(string classes[])
{
	int cameraWidth = 480;
	int dst_w = int(cameraWidth / 5.0);
	int dst_h = int(cameraWidth / 5.0 * 4 / 3);
	cv::Mat padding = cv::Mat::zeros(cv::Size(cameraWidth, dst_h * 2), CV_8UC3);
	cv::Mat image_gesture;
	for (int idx = 0; idx < 10; idx++)
	{
		image_gesture = cv::imread("./gestures/" + classes[idx]+".jpg");
		cv::resize(image_gesture, image_gesture, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);

		if(idx<5)
		{
			cv::Rect roi(idx*dst_w, 0, dst_w, dst_h);
			image_gesture.copyTo(padding(roi));
		}
		else
		{
			cv::Rect roi((idx - 5)*dst_w, dst_h,  dst_w, dst_h);
			image_gesture.copyTo(padding(roi));
		}
	}
	return padding;
}

void* __stdcall handLandmarks_Init(const char* p_palmDetModel, const char* p_handKptsDetModel, const char* p_handGesRecModel, const char* p_anchorFile, int* image_shape, const char* root_path)
{
	cv::Mat padding = load_gestureImg(gesture_classes);
	string save_root_path = root_path;
	// file vars
	string anchorFile = p_anchorFile;
	string palmModel = p_palmDetModel;
	string handModel = p_handKptsDetModel;
	string gesModel = p_handGesRecModel;

	wstring palmModelw = wstring(palmModel.begin(), palmModel.end());
	wstring handModelw = wstring(handModel.begin(), handModel.end());
	wstring gesModelw = wstring(gesModel.begin(), gesModel.end());

	const wchar_t *palmModelPath = palmModelw.c_str();
	const wchar_t *handModelPath = handModelw.c_str();
	const wchar_t *gesModelPath = gesModelw.c_str();

	// opencv vars
	int rawHeight, rawWidth, cropWidthLowBnd, cropWidth, cropHeightLowBnd, cropHeight;
	cv::Mat frame, rawFrame, showFrame, cropFrame, inFrame, tmpFrame;
	cv::VideoCapture cap;
	deque<detRect> cropHands;
	deque<detMeta> handMetaForward;

	// libtorch vars
	torch::NoGradGuard no_grad; // Disable back-grad buffering
	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	auto anchors = torch::empty({ numAnchors, 4 }, options);
	auto detectionBoxes = torch::empty({ batchSizePalm, numAnchors, outDim });
	torch::Tensor inputRawTensor, inputTensor, rawBoxesP, rawScoresP, rawScoresH, rawKeyptsH;
	deque<torch::Tensor> rawDetections;
	deque<vector<torch::Tensor>> outDetections;

	/* ---- load anchor binary file ---- */
	fstream fin(anchorFile, ios::in | ios::binary);
	fin.read((char *)anchors.data_ptr(), anchors.numel() * sizeof(float));

	/* ---- init ONNX rt ---- */
	Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	Ort::Session* sessionPalm = new Ort::Session(*env, palmModelPath, session_options);
	Ort::Session* sessionHand = new Ort::Session(*env, handModelPath, session_options);
	Ort::Session* sessionGes = new Ort::Session(*env, gesModelPath, session_options);

	//Ort::AllocatorWithDefaultOptions allocator;
	std::vector<int64_t> palm_input_node_dims = { batchSizePalm, 3, modelHeight, modelWidth };
	size_t palm_input_tensor_size = batchSizePalm * 3 * modelHeight * modelWidth;
	std::vector<int64_t> hand_input_node_dims = { batchSizeHand, 3, modelHeight, modelWidth };
	size_t hand_input_tensor_size = batchSizeHand * 3 * modelHeight * modelWidth;
	std::vector<float> input_tensor_values(palm_input_tensor_size);
	std::vector<const char *> input_node_names = { "input" };
	std::vector<const char *> output_node_names = { "output1", "output2" };
	//
	

	handLandmarks* handLdks = new handLandmarks();
	handLdks->sessions.push_back(sessionPalm);
	handLdks->sessions.push_back(sessionHand);
	handLdks->sessions.push_back(sessionGes);
	handLdks->anchors = anchors;
	handLdks->image_gestures = padding;
	handLdks->save_name = save_root_path;
	/*Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	inputTensor = torch::randn({ batchSizePalm, 3, modelHeight, modelWidth });
	Ort::Value inputTensor_ort = Ort::Value::CreateTensor<float>(memory_info, (float_t*)inputTensor.data_ptr(), palm_input_tensor_size, palm_input_node_dims.data(), 4);
	auto output_tensors = handLdks->sessions[0]->Run(Ort::RunOptions(nullptr), input_node_names.data(), &inputTensor_ort, 1, output_node_names.data(), 2);
	cout << "successful load palm detection network" << endl;*/

	return handLdks;
}



int __stdcall handLandmarks_inference(void* p_self, void* image, int* image_shape, float* output, int label_gesture, const char* name_saveImage, bool flag_record)//bool first_in, bool bbox_reinit)
{
	string gesture_name = gesture_classes[label_gesture];
	string image_suffix = name_saveImage;
	static handLandmarks* handLdks = (handLandmarks*)(p_self);
	static Ort::Session* sess_palmDet = handLdks->sessions[0];
	static Ort::Session* sess_handKptsDet = handLdks->sessions[1];
	static Ort::Session* sess_handGesRec = handLdks->sessions[2];

	unsigned char* _input = (unsigned char*)(image);
	int img_h = image_shape[0];
	int img_w = image_shape[1];
	// convert unsigned char* to cv::Mat
	cv::Mat rawFrame(img_h, img_w, CV_8UC3, _input);


	//static configuration
	static Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	static int modelWidth = 256, modelHeight = 256, modelWidth_GesRec = 32, modelHeight_GesRec = 32;
	static size_t palm_input_tensor_size = batchSizePalm * 3 * modelHeight * modelWidth;
	static std::vector<int64_t> palm_input_node_dims = { batchSizePalm, 3, modelHeight, modelWidth };
	static std::vector<const char *> input_node_names = { "input" };
	static std::vector<const char *> output_node_names = { "output1", "output2" };
	static std::vector<const char *> input_node_name_GesRec = { "input" };
	static std::vector<const char *> output_node_name_GesRec = { "output" };
	
	// queue of paml bbox and hand region
	static deque<detRect> cropHands;
	static deque<detMeta> handMetaForward;
	static ofstream record_log;
	cv::Mat frame, cropFrame, inFrame, tmpFrame;
	torch::Tensor inputRawTensor, inputTensor, rawBoxesP, rawScoresP;
	int rawHeight, rawWidth, cropHeightLowBnd, cropWidthLowBnd, cropHeight, cropWidth;

	cv::flip(rawFrame, frame, +1);
	rawHeight = frame.rows;
	rawWidth = frame.cols;
	// cropping long edge
	if (rawHeight > rawWidth)
	{
		cropHeightLowBnd = (rawHeight - rawWidth) / 2;
		cropWidthLowBnd = 0;
		cropHeight = cropWidth = rawWidth;
	}
	else
	{
		cropWidthLowBnd = (rawWidth - rawHeight) / 2;
		cropHeightLowBnd = 0;
		cropHeight = cropWidth = rawHeight;
	}
	int showHeight = cropHeight, showWidth = cropWidth;
	cv::Rect ROI(cropWidthLowBnd, cropHeightLowBnd, cropWidth, cropHeight);
	cropFrame = frame(ROI);
	//cv::imshow("crop frame", cropFrame);
	
	/* --------------------------------------- perform palm detection ------------------------------------- */
	if (handMetaForward.empty())
	{
		resize(cropFrame, inFrame, cv::Size(modelWidth, modelHeight), 0, 0, cv::INTER_LINEAR);
		cv::waitKey(1);
		cv::cvtColor(inFrame, inFrame, cv::COLOR_BGR2RGB);
		inFrame.convertTo(inFrame, CV_32F);
		inFrame = inFrame / 127.5 - 1.0;
		/* ---- Palm Detection NN Inference  ---- */
		inputRawTensor = torch::from_blob(inFrame.data, {modelHeight, modelHeight, 3});
		inputTensor = inputRawTensor.permute({ 2, 0 ,1 }).to(torch::kCPU, false, true);
		Ort::Value inputTensor_ort = Ort::Value::CreateTensor<float>(memory_info, (float_t*)inputTensor.data_ptr(), palm_input_tensor_size, palm_input_node_dims.data(), 4);
		auto output_tensors = sess_palmDet->Run(Ort::RunOptions(nullptr), input_node_names.data(), &inputTensor_ort, 1, output_node_names.data(), 2);
		float* rawBoxesPPtr = output_tensors[0].GetTensorMutableData<float>(); // bounding box
		float* rawScoresPPtr = output_tensors[1].GetTensorMutableData<float>(); // confidence
		rawBoxesP = torch::from_blob(rawBoxesPPtr, { batchSizePalm, numAnchors, outDim });
		rawScoresP = torch::from_blob(rawScoresPPtr, { batchSizePalm, numAnchors });
		
		/* ------ Tensor to Detection ------ */
		torch::Tensor detectionBoxes = torch::empty({ batchSizePalm, numAnchors, outDim });
		decodeBoxes(rawBoxesP, handLdks->anchors, detectionBoxes);
		auto detectionScores = rawScoresP.clamp(-scoreClipThrs, scoreClipThrs).sigmoid();
		
		auto mask = detectionScores >= minScoreThrs;  // palm detection confidence
		//cout << mask.sum() << endl;
		for (int i = 0; i < batchSizePalm; i++)
		{
			auto boxes = detectionBoxes[i].index(mask[i]);
			auto scores = detectionScores[i].index(mask[i]).unsqueeze(-1);
			/* ---------- NMS ----------*/
			auto outDet = weightedNMS(torch::cat({ boxes, scores }, -1));
			for (auto &det_t : outDet)
			{
				auto det = det_t.accessor<float, 1>();
				auto ymin = det[0] * showHeight;
				auto xmin = det[1] * showWidth;
				auto ymax = det[2] * showHeight;
				auto xmax = det[3] * showWidth;
				cv::Point2f handUp = cv::Point2f(det[4 + palmUpId * 2], det[4 + palmUpId * 2 + 1]),
					handDown = cv::Point2f(det[4 + palmDownId * 2], det[4 + palmDownId]);
				handMetaForward.push_back(detMeta(xmin, ymin, xmax, ymax, handUp, handDown, 0));
			}
		}
	}
	while(!handMetaForward.empty())
	{
		cropHands.push_back(handMetaForward.front().getTransformedRect(cropFrame));
		handMetaForward.pop_front();
	}

	/* ----------------- Hand Keypoint Detection NN Inference ---------------------- */
	cv::Mat showFrame = cropFrame;
	cv::Mat cropImage_Affine;
	int batchSizeHand = cropHands.size();
	if (batchSizeHand)
	{
		std::vector<int64_t> hand_input_node_dims = { batchSizeHand, 3, modelHeight, modelWidth };
		size_t hand_input_tensor_size = batchSizeHand * 3 * modelHeight * modelWidth;
		auto handsTensor = torch::empty({ batchSizeHand, modelWidth, modelHeight, 3 });
		int idx = 0;
		for (auto &cropHand : cropHands)
		{
			resize(cropHand.img, tmpFrame, cv::Size(modelWidth, modelHeight), 0, 0, cv::INTER_LINEAR);
			resize(cropHand.img, cropImage_Affine, cv::Size(modelWidth, modelHeight), 0, 0, cv::INTER_LINEAR);
			//cv::imshow("crop", cropImage_Affine);
			cv::cvtColor(tmpFrame, tmpFrame, cv::COLOR_BGR2RGB);
			tmpFrame.convertTo(tmpFrame, CV_32F);
			tmpFrame = tmpFrame / 127.5 - 1.0;
			auto tmpHand = torch::from_blob(tmpFrame.data, { 1, modelHeight, modelWidth, 3 });
			handsTensor.slice(0, idx, idx + 1) = tmpHand;
			idx++;
		}
		inputTensor = handsTensor.permute({ 0, 3, 1, 2 }).to(torch::kCPU, false, true);
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
			(float_t *)inputTensor.data_ptr(), hand_input_tensor_size, hand_input_node_dims.data(), 4);
		auto output_tensors = sess_handKptsDet->Run(Ort::RunOptions{ nullptr },
			input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
		float *rawKeyptsHPtr = output_tensors[0].GetTensorMutableData<float>();
		float *rawScoresHPtr = output_tensors[1].GetTensorMutableData<float>();

		/* ---- Draw Hand landmarks ---- */
		/* ---- Hand Gesture Recognition ---- */
		size_t memOffset = numKeypointsHand * 3;
		for (int i = 0; i < batchSizeHand; i++)
		{
			if (rawScoresHPtr[i] > handThrs) // hand keypoint detection confidence
			{

				auto tmpWidth = cropHands.front().img.cols, tmpHeight = cropHands.front().img.rows;
				cv::Mat_<float> keypointsHand = cv::Mat(numKeypointsHand, 3, CV_32F, (void *)(rawKeyptsHPtr + i * memOffset));
				float x_offset = cropHands.front().center.x - tmpWidth * 0.5,
					y_offset = cropHands.front().center.y - tmpHeight * 0.5;
				keypointsHand = keypointsHand(cv::Rect(0, 0, 2, numKeypointsHand)).t();
				keypointsHand.row(0) = keypointsHand.row(0) * tmpWidth / modelWidth + x_offset;
				keypointsHand.row(1) = keypointsHand.row(1) * tmpHeight / modelHeight + y_offset;
				auto keypointsMatRe = computePointAffine(keypointsHand, cropHands.front().affineMat, true);
				float xmin, ymin, xmax, ymax;
				xmin = xmax = keypointsMatRe(0, 0);
				ymin = ymax = keypointsMatRe(1, 0);
				int k = 0;
				float kpts_hand[21*3];
				for (int j = 0; j < numKeypointsHand; j++)
				{
					kpts_hand[3 * j] = keypointsMatRe(0, j);
					kpts_hand[3 * j + 1] = keypointsMatRe(1, j);
					kpts_hand[3 * j + 2] = keypointsMatRe(2, j);
					cv::circle(showFrame, cv::Point2f(keypointsMatRe(0, j), keypointsMatRe(1, j)), 4, { 255, 0, 0 }, -1);
					cv::circle(cropHands.front().img, cv::Point2f(keypointsHand(0, j) - x_offset + cropHands.front().affineMat.at<float>(0, 2), keypointsHand(1, j) - y_offset + cropHands.front().affineMat.at<float>(1, 2)), 2, { 0, 255, 0 });
					if (nonFingerId[k] == j)
					{
						xmin = min(xmin, keypointsMatRe(0, j));
						xmax = max(xmax, keypointsMatRe(0, j));
						ymin = min(ymin, keypointsMatRe(1, j));
						ymax = max(ymax, keypointsMatRe(1, j));
						k++;
					}
				}
				// cv::imshow("PalmDetection", cropHands.front().img);
				// cv::waitKey();
				for (int j = 0; j < numJointConnect; j++)
				{
					cv::line(showFrame, cv::Point2f(keypointsMatRe(0, jointConnect[j][0]), keypointsMatRe(1, jointConnect[j][0])),
						cv::Point2f(keypointsMatRe(0, jointConnect[j][1]), keypointsMatRe(1, jointConnect[j][1])), { 255, 255, 255 }, 2);
				}
				auto handUp = cv::Point2f(keypointsMatRe(0, handUpId), keypointsMatRe(1, handUpId));
				auto handDown = cv::Point2f(keypointsMatRe(0, handDownId), keypointsMatRe(1, handDownId));
				//handMetaForward.push_back(detMeta(xmin, ymin, xmax, ymax, handUp, handDown, 1));  //tracking by landmarks

				// show or record
				if (rawScoresHPtr[i] > handThrs + 0.1)
				{
					//cv::imshow("full region", cropHands.front().image_fullRegion);
					cv::imshow("hand region", cropHands.front().image_handRegion);
					//if(flag_record)
					//{
					//	/*string imageFile_fullRegion = handLdks->save_name + "/image_fullRegion/" + gesture_name + "/" + image_suffix + ".jpg";
					//	string imageFile_handRegion = handLdks->save_name + "/image_handRegion/" + gesture_name + "/" + image_suffix + ".jpg";
					//	ofstream record_log(handLdks->save_name + "/datalist.txt", ios::app);
					//	record_log << gesture_name << "\t" << imageFile_fullRegion << "\t" << imageFile_handRegion << "\t";
					//	for (int idx = 0; idx < 4; idx++)
					//	{
					//		record_log << cropHands.front().palm_bbox[idx] << ",";
					//	}
					//	record_log << "\t";
					//	for (int idx = 0; idx < 4; idx++)
					//	{
					//		record_log << cropHands.front().hand_bbox[idx] << ",";
					//	}
					//	record_log << "\t";
					//	record_log << cropHands.front().center.x << "," << cropHands.front().center.y << "\t" << cropHands.front().rotate_angle<< "\t";
					//	for (int idx = 0; idx < numKeypointsHand * 3; idx++)
					//	{
					//		record_log << kpts_hand[idx] << ",";
					//	}
					//	record_log << "\n";
					//	record_log.close();
					//	cv::imwrite(imageFile_fullRegion, cropHands.front().image_fullRegion);
					//	cv::imwrite(imageFile_handRegion, cropHands.front().image_handRegion);*/
					//}

					auto cropHandTensor = torch::empty({ 1, modelWidth_GesRec, modelHeight_GesRec, 3 });
					/* ---------  padding and resize  --------- */
					auto h_tmpCropFrame = cropHands.front().image_handRegion.rows;
					auto w_tmpCropFrame = cropHands.front().image_handRegion.cols;
					auto long_side = max(h_tmpCropFrame, w_tmpCropFrame);
					cv::Mat tmpCropFrame = cv::Mat::zeros(cv::Size(long_side, long_side), CV_8UC3);
					if (h_tmpCropFrame > w_tmpCropFrame) // height > width
					{
						cv::Rect region(int(0.5*(long_side - w_tmpCropFrame)), 0, w_tmpCropFrame, h_tmpCropFrame);
						cropHands.front().image_handRegion.copyTo(tmpCropFrame(region));
					}
					else
					{
						cv::Rect region(0, int(0.5*(long_side - h_tmpCropFrame)), w_tmpCropFrame, h_tmpCropFrame);
						cropHands.front().image_handRegion.copyTo(tmpCropFrame(region));
					}
					tmpCropFrame.convertTo(tmpCropFrame, CV_32F);
					cv::resize(tmpCropFrame, tmpCropFrame, cv::Size(32, 32), 0, 0, cv::INTER_LINEAR);
					//tmpCropFrame /= 255.0;
					//cout << tmpCropFrame << endl;
					auto tmpCropHand = torch::from_blob(tmpCropFrame.data, { 1, modelHeight_GesRec, modelWidth_GesRec, 3 });
					// normalized
					/*tmpCropHand.slice(3, 0, 1) = (tmpCropHand.slice(3, 0, 1) - 0.4914) / 0.247;
					tmpCropHand.slice(3, 1, 2) = (tmpCropHand.slice(3, 1, 2) - 0.4822) / 0.243;
					tmpCropHand.slice(3, 2, 3) = (tmpCropHand.slice(3, 2, 3) - 0.4465) / 0.261;*/
					// cout << tmpCropHand << endl;
					tmpCropHand = tmpCropHand.permute({ 0, 3, 1, 2 }).to(torch::kCPU, false, true);
					std::vector<int64_t> cropHand_input_node_dims = { 1, 3, modelHeight_GesRec, modelWidth_GesRec };
					size_t cropHand_input_tensor_size = 1 * 3 * modelHeight_GesRec * modelWidth_GesRec;
					Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
						(float_t *)tmpCropHand.data_ptr(), cropHand_input_tensor_size, cropHand_input_node_dims.data(), 4);
					auto output_tensors = sess_handGesRec->Run(Ort::RunOptions{ nullptr },
						input_node_name_GesRec.data(), &input_tensor, 1, output_node_name_GesRec.data(), 1);
					float *GesOutput = output_tensors[0].GetTensorMutableData<float>();
					cv::Mat_<float> GesScores = cv::Mat(10, 1, CV_32F, (void *)(GesOutput));
					// cout << GesScores << endl;
				}
			}
			cropHands.pop_front();
		}
	}
	cv::Mat image_gestures_copy;
	if(label_gesture<5)
	{
		cv::Rect selected_gesture(label_gesture*handLdks->image_gestures.cols/5, 0, handLdks->image_gestures.cols / 5, handLdks->image_gestures.rows / 2);
		
		handLdks->image_gestures.copyTo(image_gestures_copy);
		cv::rectangle(image_gestures_copy, selected_gesture, { 0.0, 255.0, 0.0 }, 3);
	}
	else
	{
		cv::Rect selected_gesture((label_gesture-5)*handLdks->image_gestures.cols / 5, handLdks->image_gestures.rows / 2, handLdks->image_gestures.cols / 5, handLdks->image_gestures.rows / 2);
		handLdks->image_gestures.copyTo(image_gestures_copy);
		cv::rectangle(image_gestures_copy, selected_gesture, { 0.0, 255.0, 0.0 }, 3);
	}
	cv::vconcat(image_gestures_copy, showFrame, showFrame);
	cv::imshow("handDetection", showFrame);
	if (cv::waitKey(5) == 27)
	{
		cv::destroyAllWindows();
		return 1;
	};
	if (!videoMode)
	{
		cv::waitKey(1);
	}
	return 0;
}


