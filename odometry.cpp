#include <iostream>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;



struct FeatureSet 
{
    std::vector<cv::Point2f> points;
    std::vector<int> ages;


    int size()
    {
        return points.size();
    }

    void clear()
    {
        points.clear();
        ages.clear();
    }

};


class Bucket
{
    public:
        int id;
        int max_size;

        FeatureSet features;

        Bucket(int size)
        {
            max_size = size;
        }
        ~Bucket(){}

        void add_feature(cv::Point2f point, int age)
        {
            // won't add feature with age > 10;
            int age_threshold = 10;
            if (age < age_threshold)
            {
                // insert any feature before bucket is full
                if (size()<max_size)
                {
                    features.points.push_back(point);
                    features.ages.push_back(age);

                }
                else
                // insert feature with old age and remove youngest one
                {
                    int age_min = features.ages[0];
                    int age_min_idx = 0;

                    for (int i = 0; i < size(); i++)
                    {
                        if (age < age_min)
                        {
                            age_min = age;
                            age_min_idx = i;
                        }
                    }
                    features.points[age_min_idx] = point;
                    features.ages[age_min_idx] = age;
                }
            }

        }
        void get_features(FeatureSet &current_features)
        {
            current_features.points.insert(current_features.points.end(), features.points.begin(), features.points.end());
            current_features.ages.insert(current_features.ages.end(), features.ages.begin(), features.ages.end());

        }

        int size()
        {
            return features.points.size();
        }
    
};



void detect_features(cv::Mat &img, std::vector<cv::Point2f> &points)
/*detect FAST features (corners) in an input image and returns 
a vector of 2D points*/
{
    std::vector<cv::KeyPoint> keypoints;
    int treshold = 20;
    bool nonmaxSuppression = true;

    cv::FAST(img, keypoints, treshold, nonmaxSuppression);
    cv::KeyPoint::convert(keypoints, points, /*mask */std::vector<int>());
}

void append_new_features(cv::Mat &img, FeatureSet &currentFeatures)
{
    std::vector<cv::Point2f> newPoints;
    detect_features(img, newPoints);
    currentFeatures.points.insert(currentFeatures.points.end(), newPoints.begin(), newPoints.end());
    std::vector<int> newAges(newPoints.size(), 0);
    currentFeatures.ages.insert(currentFeatures.ages.end(), newAges.begin(), newAges.end());
}


void delete_unmatched_features(std::vector<cv::Point2f> &leftPoints_t0, std::vector<cv::Point2f> &rightPoints_t0,
                                 std::vector<cv::Point2f> &leftPoints_t1, std::vector<cv::Point2f> &rightPoints_t1,
                                 std::vector<uchar> &status0, std::vector<uchar> &status1,  std::vector<uchar> &status2,
                                 std::vector<uchar> &status3, std::vector<int> &ages, std::vector<cv::Point2f> &returnedLeftPoints_t0)
{
    for (int i=0; i < ages.size(); i++)
    {
        ages[i] += 1;
    }
    int indexCorrection = 0;
    for (int i =0; i < status3.size(); i++)
    {

        cv::Point2f pt0 = leftPoints_t0.at(i-indexCorrection);
        cv::Point2f pt1 = leftPoints_t1.at(i-indexCorrection);
        cv::Point2f pt2 = rightPoints_t0.at(i-indexCorrection);
        cv::Point2f pt3 = rightPoints_t1.at(i-indexCorrection);
        cv::Point2f pt0_returned = returnedLeftPoints_t0.at(i-indexCorrection);
        if ((status3.at(i) == 0)||(pt3.x<0)||(pt3.y<0)||
            (status2.at(i) == 0)||(pt2.x<0)||(pt2.y<0)||
            (status1.at(i) == 0)||(pt1.x<0)||(pt1.y<0)||
            (status0.at(i) == 0)||(pt0.x<0)||(pt0.y<0))
            {
                if((pt0.x<0)||(pt0.y<0)||(pt1.x<0)||(pt1.y<0)||(pt2.x<0)||(pt2.y<0)||(pt3.x<0)||(pt3.y<0))    
                    {
                        status3.at(i) = 0;
                    }
                leftPoints_t0.erase (leftPoints_t0.begin() + (i - indexCorrection));
                leftPoints_t1.erase (leftPoints_t1.begin() + (i - indexCorrection));
                rightPoints_t0.erase (rightPoints_t0.begin() + (i - indexCorrection));
                rightPoints_t1.erase (rightPoints_t1.begin() + (i - indexCorrection));
                returnedLeftPoints_t0.erase (returnedLeftPoints_t0.begin() + (i - indexCorrection));

                ages.erase (ages.begin() + (i - indexCorrection));
                indexCorrection++;
            }

    }
}


void circular_matching(cv::Mat &leftImage_t0, cv::Mat &rightImage_t0, cv::Mat &leftImage_t1, cv::Mat &rightImage_t1, std::vector<cv::Point2f> &leftPoints_t0, 
std::vector<cv::Point2f> &rightPoints_t0, std::vector<cv::Point2f> &leftPoints_t1, 
std::vector<cv::Point2f> &rightPoints_t1, FeatureSet &currentFeatures, std::vector<cv::Point2f> &returnedLeftPoints_t0)
{
    std::vector<float> err;                    
    cv::Size winSize=cv::Size(21,21);                                                                                             
    cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);

    std::vector<uchar> status0;
    std::vector<uchar> status1;
    std::vector<uchar> status2;
    std::vector<uchar> status3;


    calcOpticalFlowPyrLK(leftImage_t0, rightImage_t0, leftPoints_t0, rightPoints_t0, status0, err, winSize, 3, termcrit, 0, 0.001);
    calcOpticalFlowPyrLK(rightImage_t0, rightImage_t1, rightPoints_t0, rightPoints_t1, status1, err, winSize, 3, termcrit, 0, 0.001);
    calcOpticalFlowPyrLK(rightImage_t1, leftImage_t1, rightPoints_t1, leftPoints_t1, status2, err, winSize, 3, termcrit, 0, 0.001);
    calcOpticalFlowPyrLK(leftImage_t1, leftImage_t0, leftPoints_t1, returnedLeftPoints_t0, status3, err, winSize, 3, termcrit, 0, 0.001);

    //std::cout << "returnedLeftPoints_t0 size : " << returnedLeftPoints_t0.size() << std::endl;

    delete_unmatched_features(leftPoints_t0, rightPoints_t0, leftPoints_t1, rightPoints_t1, status0, status1, status2, status3, currentFeatures.ages, returnedLeftPoints_t0);
    //std::cout << "returnedLeftPoints_t0 size inside circular matching: " << returnedLeftPoints_t0.size() << std::endl;
}



void check_valid_match(std::vector<cv::Point2f> &points, std::vector<cv::Point2f> &points_return, std::vector<bool> &status, int threshold)
{
    int offset;
    for (int i = 0; i < points.size(); i++)
    {
        offset = std::max(std::abs(points[i].x - points_return[i].x), std::abs(points[i].y - points_return[i].y));
        // std::cout << offset << ", ";

        if(offset > threshold)
        {
            status.push_back(false);
        }
        else
        {
            status.push_back(true);
        }
    }
}

void remove_invalid_points(std::vector<cv::Point2f>& points, const std::vector<bool>& status)
{
    int index = 0;
    for (int i = 0; i < status.size(); i++)
    {
        if (status[i] == false)
        {
            points.erase(points.begin() + index);
        }
        else
        {
            index ++;
        }
    }
}



void bucketing_features(cv::Mat& image, FeatureSet& current_features, int bucket_size, int features_per_bucket)
{
// This function buckets features
// image: only use for getting dimension of the image
// bucket_size: bucket size in pixel is bucket_size*bucket_size
// features_per_bucket: number of selected features per bucket
    int image_height = image.rows;
    int image_width = image.cols;
    int buckets_nums_height = image_height/bucket_size;
    int buckets_nums_width = image_width/bucket_size;
    int buckets_number = buckets_nums_height * buckets_nums_width;

    std::vector<Bucket> Buckets;

    // initialize all the buckets
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
      for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
      {
        Buckets.push_back(Bucket(features_per_bucket));
      }
    }

    // bucket all current features into buckets by their location
    int buckets_nums_height_idx, buckets_nums_width_idx, buckets_idx;
    for (int i = 0; i < current_features.points.size(); ++i)
    {
      buckets_nums_height_idx = current_features.points[i].y/bucket_size;
      buckets_nums_width_idx = current_features.points[i].x/bucket_size;
      buckets_idx = buckets_nums_height_idx*buckets_nums_width + buckets_nums_width_idx;
      Buckets[buckets_idx].add_feature(current_features.points[i], current_features.ages[i]);

    }

    // get features back from buckets
    current_features.clear();
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
      for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
      {
         buckets_idx = buckets_idx_height*buckets_nums_width + buckets_idx_width;
         Buckets[buckets_idx].get_features(current_features);
      }
    }

    //std::cout << "current features number after bucketing: " << current_features.size() << std::endl;

}


void matching_features(FeatureSet &currentFeatures, cv::Mat &leftImage_t0, cv::Mat &rightImage_t0, cv::Mat &leftImage_t1, cv::Mat &rightImage_t1, 
std::vector<cv::Point2f> &leftPoints_t0, std::vector<cv::Point2f> &rightPoints_t0, std::vector<cv::Point2f> &leftPoints_t1, std::vector<cv::Point2f> &rightPoints_t1)
{

    std::vector<cv::Point2f> returnedLeftPoints_t0;

    if (currentFeatures.size() < 2000)
    {
        append_new_features(leftImage_t0, currentFeatures);
    }

    //std::cout << "current features size after appending : " << currentFeatures.size() << std::endl;

    int bucket_size = leftImage_t0.rows/10;
    int features_per_bucket = 1;
    bucketing_features(leftImage_t0, currentFeatures, bucket_size, features_per_bucket);

    leftPoints_t0 = currentFeatures.points;
    
    circular_matching(leftImage_t0, rightImage_t0, leftImage_t1, rightImage_t1, leftPoints_t0,  rightPoints_t0, leftPoints_t1, rightPoints_t1, currentFeatures, returnedLeftPoints_t0);

    //std::cout << "current returned left points size : " << returnedLeftPoints_t0.size() << std::endl;

    std::vector<bool> status;
    check_valid_match(leftPoints_t0, returnedLeftPoints_t0, status, 0);

    remove_invalid_points(leftPoints_t0, status);
    remove_invalid_points(leftPoints_t1, status);
    remove_invalid_points(rightPoints_t0, status);
    remove_invalid_points(rightPoints_t1, status);

    currentFeatures.points = leftPoints_t1;
}




void display_tracking(cv::Mat &leftImage_t1, 
                     std::vector<cv::Point2f> &leftPoints_t0,
                     std::vector<cv::Point2f> &leftPoints_t1)
{
      // -----------------------------------------
      // Display feature racking
      // -----------------------------------------
      int radius = 5;
      cv::Mat vis;

      cv::cvtColor(leftImage_t1, vis, cv::COLOR_GRAY2BGR, 3);


      //std::cout << "display leftPoints_t0 size : " << leftPoints_t0.size() << std::endl;
      //std::cout << "display leftPoints_t1 size : " << leftPoints_t1.size() << std::endl;


      for (int i = 0; i < leftPoints_t0.size(); i++)
      {
          cv::circle(vis, cv::Point(leftPoints_t0[i].x, leftPoints_t0[i].y), radius, CV_RGB(0,255,0));
      }

      for (int i = 0; i < leftPoints_t1.size(); i++)
      {
          cv::circle(vis, cv::Point(leftPoints_t1[i].x, leftPoints_t1[i].y), radius, CV_RGB(255,0,0));
      }

      for (int i = 0; i < leftPoints_t1.size(); i++)
      {
          cv::line(vis, leftPoints_t0[i], leftPoints_t1[i], CV_RGB(0,255,0));
      }

      cv::imshow("vis ", vis ); 
      cv::waitKey(1); 
}



void bundle_adjustment()
{


}









/* function that loads images from a given path*/
void load_image(int &frameId, std::vector<std::string> &imageNames, cv::Mat &img)
{
    img = cv::imread(imageNames[frameId], cv::IMREAD_GRAYSCALE);
}





void read_images_paths(std::string &path, 
std::vector<std::string> &RightImageNames, std::vector<std::string> &LeftImageNames)
/*
    function to read images from kitti Dataset, 
    image_0 refers to left camera while image_1 refers to right camera.
*/
{
    /*
        scan and sort the image names in the directory
        and then append 2 std::vector's with the sorted paths
    */
    for (const auto &entry : fs::directory_iterator(path+"/image_0"))
    {
        LeftImageNames.push_back(entry.path().string());
    }

    for (const auto &entry: fs::directory_iterator(path+"/image_1"))
    {
        RightImageNames.push_back(entry.path().string());
    }

    std::sort(RightImageNames.begin(), RightImageNames.end());
    std::sort(LeftImageNames.begin(), LeftImageNames.end());
}


int main()
{

    std::vector<std::string> rightImageNames, leftImageNames;
    std::string path = "../data";

    read_images_paths(path, rightImageNames, leftImageNames);

    //std::cout << rightImageNames[1] << std::endl;
    //std::cout << leftImageNames[1] << std::endl;
    

    /*
        initialize the first pair of images leftImageT0, rightImageT0
    */
   cv::Mat leftImage_t0, rightImage_t0;
   cv::Mat leftImage_t1, rightImage_t1;





   FeatureSet currentFeatures;
   int frameId = 0;

    //loading right image
   load_image(frameId, rightImageNames, rightImage_t0);

   //loading left image
   load_image(frameId, leftImageNames, leftImage_t0);


   /*loop over left and right frames*/

   for (int i = frameId + 1; frameId < rightImageNames.size(); frameId++)
   {
        std::cout << "frame number: " << frameId+1<< std::endl;
        //loading right image
        load_image(frameId, rightImageNames, rightImage_t1);

        //loading left image
        load_image(frameId, leftImageNames, leftImage_t1);

        std::vector<cv::Point2f> leftPoints_t0, rightPoints_t0, leftPoints_t1, rightPoints_t1;

        matching_features(currentFeatures, leftImage_t0, rightImage_t0, leftImage_t1, rightImage_t1, 
        leftPoints_t0, rightPoints_t0, leftPoints_t1, rightPoints_t1);


        /*end of iteration*/
        leftImage_t0 = leftImage_t1;
        rightImage_t0 = rightImage_t1;

        //std::cout << "inside of main : leftPoints_t0.size(): " << leftPoints_t0.size() << std::endl;


        display_tracking(leftImage_t1, leftPoints_t0, leftPoints_t1);

   }
}