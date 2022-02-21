#pragma once
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <functional>
#include <chrono>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <onnxruntime_cxx_api.h>
// #ifdef CV_CXX11
#include <mutex>
#include <thread>
#include <queue>


using namespace std;

int modelWidth = 256, modelHeight = 256, modelWidth_GesRec = 32, modelHeight_GesRec = 32;
int numAnchors = 2944, outDim = 18, batchSizePalm = 1, batchSizeHand=1;
int numKeypointsPalm = 7, numKeypointsHand = 21, numJointConnect = 20, palmDetFreq = 20;
float scoreClipThrs = 100.0, minScoreThrs = 0.80, minSuppressionThrs = 0.3, handThrs = 0.8;
float palm_shift_y = 0.5, palm_shift_x = 0, palm_box_scale = 2.6,
hand_shift_y = 0, hand_shift_x = 0, hand_box_scale = 2.1;

bool videoMode = true, showPalm = false;

int jointConnect[20][2] = {
	{0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 5}, {5, 6}, {6, 7}, {7, 8}, {0, 9}, {9, 10}, {10, 11}, {11, 12}, {0, 13}, {13, 14}, {14, 15}, {15, 16}, {0, 17}, {17, 18}, {18, 19}, {19, 20} };
int nonFingerId[] = {
	0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18 };
int handUpId = 9, handDownId = 0, palmUpId = 2, palmDownId = 0;

string gesture_classes[] = { "number_1", "number_2", "number_3", "another_number_3", "number_4", "number_5", "number_6", "thumb_up", "ok", "heart" };

cv::Mat cropResize(const cv::Mat &frame, int xMin, int yMin, int xCrop, int yCrop);
inline void matToTensor(cv::Mat const &src, torch::Tensor &out);
void decodeBoxes(const torch::Tensor &rawBoxesP, const torch::Tensor &anchors, torch::Tensor &boxes);
vector<torch::Tensor> NMS(const torch::Tensor &detections);
vector<torch::Tensor> weightedNMS(const torch::Tensor &detections);
torch::Tensor computeIoU(const torch::Tensor &boxA, const torch::Tensor &boxB);
cv::Mat_<float> computePointAffine(cv::Mat_<float> &pointsMat, cv::Mat_<float> &affineMat, bool inverse);


struct detRect
{
	cv::Mat img;
	cv::Mat_<float> affineMat;
	cv::Point2f center;
	// added for hand crop information recording
	cv::Mat image_fullRegion;
	cv::Mat image_handRegion; //after expanding
	float palm_bbox[4];
	float hand_bbox[4];
	float rotate_angle;
	detRect(cv::Mat &src, cv::Mat_<float> &a_mat, cv::Point2f &point, cv::Mat &src_fullRegion, cv::Mat &src_handRegion, cv::Rect &_palm_bbox, cv::Rect &_hand_bbox, float _angle):
		img(src), affineMat(a_mat), center(point), image_fullRegion(src_fullRegion), image_handRegion(src_handRegion),  rotate_angle(_angle)
	{
		palm_bbox[0] = _palm_bbox.x;
		palm_bbox[1] = _palm_bbox.y;
		palm_bbox[2] = _palm_bbox.width;
		palm_bbox[3] = _palm_bbox.height;
		hand_bbox[0] = _hand_bbox.x;
		hand_bbox[1] = _hand_bbox.y;
		hand_bbox[2] = _hand_bbox.width;
		hand_bbox[3] = _hand_bbox.height;
	}
};

struct detMeta
{
	int fid;     // Frame id, pay attention to MAX_INT;
	int detType; // 0 is palm det, 1 is landmark det
	int xmin, ymin, xmax, ymax;
	float shift_x, shift_y, box_scale;
	cv::Point2f handUp, handDown;
	detMeta(int x_min, int y_min, int x_max, int y_max,
		cv::Point2f &Up, cv::Point2f &Down, int type = 0, int id = 0) :
		fid(id), xmin(x_min), ymin(y_min), xmax(x_max), ymax(y_max), detType(type), handUp(Up), handDown(Down)
	{
		if (type == 0) //detection, 7 hand keypoints
		{
			shift_x = palm_shift_x;
			shift_y = palm_shift_y;
			box_scale = palm_box_scale;
		}
		else  //tracking, 21 hand keypoints
		{
			shift_x = hand_shift_x;
			shift_y = hand_shift_y;
			box_scale = hand_box_scale;
		}
	}
	detRect getTransformedRect(cv::Mat &img, bool square_long = true)
	{
		auto xscale = xmax - xmin, yscale = ymax - ymin;
		cv::Rect paml_bbox = cv::Rect(xmin, ymin, xscale, yscale);
		/* ---- Compute rotatation ---- */
		auto angleRad = atan2(handDown.x - handUp.x, handDown.y - handUp.y);
		auto angleDeg = angleRad * 180 / M_PI; // the angle between palm direction and vertical direction
		// Movement
		// shift_y > 0 : move 0(palmDownId) --> 2(palmUpId); shift_x > 0 : move right hand side of 0->2
		//(x_center, y_center): the center point after shifting but before rotating
		auto x_center = xmin + xscale * (0.5 - shift_y * sin(angleRad) + shift_x * cos(angleRad));
		auto y_center = ymin + yscale * (0.5 - shift_y * cos(angleRad) - shift_x * sin(angleRad));

		cv::Mat img_copy, img_handRegion;
		img.copyTo(img_copy);
		float box_scale_handRegion = 3.0;
		auto x_leftTop = max(int(x_center - xscale * box_scale_handRegion /2.0), 0), y_leftTop = max(int(y_center - yscale * box_scale_handRegion /2.0), 0);
		//cv::rectangle(img_copy, cv::Rect(x_leftTop, y_leftTop, min(int(xscale*2.6), img_copy.cols - x_leftTop - 1), min(int(yscale*2.6), img_copy.rows - y_leftTop - 1)), { 0.0, 255.0, 0.0 });
		// expand by 2.6 rather than box_scale(2.1/2.6)
		cv::Rect roi_hand = cv::Rect(x_leftTop, y_leftTop, min(int(xscale* box_scale_handRegion), img_copy.cols - x_leftTop - 1), min(int(yscale* box_scale_handRegion), img_copy.rows - y_leftTop - 1));
		img_handRegion = img_copy(roi_hand);
		//cv::imshow("Hand Region", img_handRegion);
		cv::waitKey(1);

		if (square_long)
			xscale = yscale = max(xscale, yscale); //padding

		auto xrescale = xscale * box_scale, yrescale = yscale * box_scale;
		/* ---- Get cropped Hands ---- */
		// affineMat.size: 2x3
		cv::Mat_<float> affineMat = cv::getRotationMatrix2D(cv::Point2f(img.cols, img.rows) / 2, -angleDeg, 1); // center, angle, scale
		auto bbox = cv::RotatedRect(cv::Point2f(), img.size(), -angleDeg).boundingRect2f();
		affineMat.at<float>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
		affineMat.at<float>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;

		cv::Mat rotFrame;
		cv::warpAffine(img, rotFrame, affineMat, bbox.size());

		//cv::imshow("Rotated", rotFrame);
		// cv::waitKey();
		// Cropping & Point Affine Transformation
		cv::Mat_<float> pointMat(2, 1);
		pointMat << x_center, y_center;
		cv::Mat_<float> rotPtMat = computePointAffine(pointMat, affineMat, false); // center point after affine transformation
		cv::Point2f rotCenter(rotPtMat(0), rotPtMat(1));
		// Out of range cases
		float xrescale_2 = xrescale / 2, yrescale_2 = yrescale / 2;
		float xDwHalf = min(rotCenter.x, xrescale_2), yDwHalf = min(rotCenter.y, yrescale_2);
		float xUpHalf = rotCenter.x + xrescale_2 > rotFrame.cols ? rotFrame.cols - rotCenter.x : xrescale_2;
		float yUpHalf = rotCenter.y + yrescale_2 > rotFrame.rows ? rotFrame.rows - rotCenter.y : yrescale_2;
		auto cropHand = rotFrame(cv::Rect(rotCenter.x - xDwHalf, rotCenter.y - yDwHalf, xDwHalf + xUpHalf, yDwHalf + yUpHalf));
		//cv::imshow("ROI", cropHand);
		cv::copyMakeBorder(cropHand, cropHand, yrescale_2 - yDwHalf, yrescale_2 - yUpHalf,
			xrescale_2 - xDwHalf, xrescale_2 - xUpHalf, cv::BORDER_CONSTANT);
		return detRect(cropHand, affineMat, rotCenter, img_copy, img_handRegion, paml_bbox, roi_hand, angleDeg);
	}
};

struct handLandmarks
{
	vector<Ort::Session*> sessions;
	torch::Tensor anchors;
	cv::Mat image_gestures;
	string save_name;
};


void decodeBoxes(const torch::Tensor &rawBoxesP, const torch::Tensor &anchors, torch::Tensor &boxes)
{
	auto x_center = rawBoxesP.slice(2, 0, 1) / modelWidth * anchors.slice(1, 2, 3) + anchors.slice(1, 0, 1);
	auto y_center = rawBoxesP.slice(2, 1, 2) / modelHeight * anchors.slice(1, 3, 4) + anchors.slice(1, 1, 2);

	auto w = rawBoxesP.slice(2, 2, 3) / modelWidth * anchors.slice(1, 2, 3);
	auto h = rawBoxesP.slice(2, 3, 4) / modelHeight * anchors.slice(1, 3, 4);

	boxes.slice(2, 0, 1) = y_center - h / 2; // ymin
	boxes.slice(2, 1, 2) = x_center - w / 2; // xmin
	boxes.slice(2, 2, 3) = y_center + h / 2; // ymax
	boxes.slice(2, 3, 4) = x_center + w / 2; // xmax

	int offset = 4 + numKeypointsPalm * 2;
	boxes.slice(2, 4, offset, 2) = rawBoxesP.slice(2, 4, offset, 2) / modelWidth * anchors.slice(1, 2, 3) + anchors.slice(1, 0, 1);
	boxes.slice(2, 5, offset, 2) = rawBoxesP.slice(2, 5, offset, 2) / modelHeight * anchors.slice(1, 3, 4) + anchors.slice(1, 1, 2);
}

vector<torch::Tensor> weightedNMS(const torch::Tensor &detections)
{
	vector<torch::Tensor> outDets;
	if (detections.size(0) == 0)
		return outDets;
	auto remaining = detections.slice(1, outDim, outDim + 1).argsort(0, true).squeeze(-1);
	// cout<<remaining.sizes()<<"  "<<remaining[0];
	// cout<<detections[remaining[0]].sizes()<<"\n";
	// torch::Tensor IoUs;
	while (remaining.size(0) > 0)
	{
		auto weightedDet = detections[remaining[0]].to(torch::kCPU, false, true);
		auto firstBox = detections[remaining[0]].slice(0, 0, 4).unsqueeze(0);
		auto otherBoxes = detections.index(remaining).slice(1, 0, 4);
		// cout<<firstBox.sizes()<<"    "<<otherBoxes.sizes();
		auto IoUs = computeIoU(firstBox, otherBoxes);
		// cout<<IoUs.sizes();
		auto overlapping = remaining.index(IoUs > minSuppressionThrs);
		remaining = remaining.index(IoUs <= minSuppressionThrs);
		if (overlapping.size(0) > 1)
		{
			auto coords = detections.index(overlapping).slice(1, 0, outDim);
			auto scores = detections.index(overlapping).slice(1, outDim, outDim + 1);
			auto totalScore = scores.sum();
			weightedDet.slice(0, 0, outDim) = (coords * scores).sum(0) / totalScore;
			weightedDet[outDim] = totalScore / overlapping.size(0);
		}
		outDets.push_back(weightedDet);
	}
	// cout<<outDets<<endl;
	return outDets;
}

vector<torch::Tensor> NMS(const torch::Tensor &detections)
{
	vector<torch::Tensor> outDets;
	if (detections.size(0) == 0)
		return outDets;
	auto remaining = detections.slice(1, outDim, outDim + 1).argsort(0, true).squeeze(-1);
	// cout<<remaining.sizes()<<"  "<<remaining[0];
	// cout<<detections[remaining[0]].sizes()<<"\n";
	// torch::Tensor IoUs;
	while (remaining.size(0) > 0)
	{
		auto Det = detections[remaining[0]].to(torch::kCPU, false, true);
		auto firstBox = detections[remaining[0]].slice(0, 0, 4).unsqueeze(0);
		auto otherBoxes = detections.index(remaining).slice(1, 0, 4);
		// cout<<firstBox.sizes()<<"    "<<otherBoxes.sizes();
		auto IoUs = computeIoU(firstBox, otherBoxes);
		// cout<<IoUs.sizes();
		auto overlapping = remaining.index(IoUs > minSuppressionThrs);
		remaining = remaining.index(IoUs <= minSuppressionThrs);

		outDets.push_back(Det);
	}
	// cout<<outDets<<endl;
	return outDets;
}

torch::Tensor computeIoU(const torch::Tensor &boxA, const torch::Tensor &boxB)
{
	// Compute Intersection
	int sizeA = boxA.size(0), sizeB = boxB.size(0);
	auto max_xy = torch::min(boxA.slice(1, 2, 4).unsqueeze(1).expand({ sizeA, sizeB, 2 }),
		boxB.slice(1, 2, 4).unsqueeze(0).expand({ sizeA, sizeB, 2 }));
	auto min_xy = torch::max(boxA.slice(1, 0, 2).unsqueeze(1).expand({ sizeA, sizeB, 2 }),
		boxB.slice(1, 0, 2).unsqueeze(0).expand({ sizeA, sizeB, 2 }));
	auto coords = (max_xy - min_xy).relu();
	auto interX = (coords.slice(2, 0, 1) * coords.slice(2, 1, 2)).squeeze(-1); // [sizeA, sizeB]

	auto areaA = ((boxA.slice(1, 2, 3) - boxA.slice(1, 0, 1)) *
		(boxA.slice(1, 3, 4) - boxA.slice(1, 1, 2)))
		.expand_as(interX);
	auto areaB = ((boxB.slice(1, 2, 3) - boxB.slice(1, 0, 1)) *
		(boxB.slice(1, 3, 4) - boxB.slice(1, 1, 2)))
		.squeeze(-1)
		.unsqueeze(0)
		.expand_as(interX);
	// cout<<areaA.sizes()<<"  "<<areaB.sizes()<<endl;
	auto unions = areaA + areaB - interX;
	return (interX / unions).squeeze(0);
}

cv::Mat_<float> computePointAffine(cv::Mat_<float> &pointsMat, cv::Mat_<float> &affineMat, bool inverse)
{
	// cout<<pointsMat.size<<endl;
	if (!inverse)
	{
		cv::Mat_<float> ones = cv::Mat::ones(pointsMat.cols, 1, CV_32F);
		pointsMat.push_back(ones);
		return affineMat * pointsMat;
	}
	else
	{
		pointsMat.row(0) -= affineMat.at<float>(0, 2);
		pointsMat.row(1) -= affineMat.at<float>(1, 2);
		cv::Mat_<float> affineMatInv = affineMat(cv::Rect(0, 0, 2, 2)).inv();
		return affineMatInv * pointsMat;
	}
}

extern "C" _declspec(dllexport) void* __stdcall handLandmarks_Init(const char* p_palmDetModel, const char* p_handKptsDetModel, 
	const char* p_handGesRecModel,const char* p_anchorFile, int* image_shape, const char* root_path);
extern "C" _declspec(dllexport) int __stdcall handLandmarks_inference(void* p_self, void* image, int* image_shape, float* output, 
	int label_gesture, const char* imagePrefix, bool flag_record); // bool first_in, bool bbox_reinit);

