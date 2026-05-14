#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <processing_bci/eeg_fbcsp.h>
#include "processing_bci/utils.hpp"
#include <vector>
#include <string>

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_publisher_slda");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string csv_filename, topic;
    double sample_rate;

    if (!pnh.getParam("csv_file",    csv_filename)) { ROS_ERROR("'csv_file' not set.");    return 1; }
    if (!pnh.getParam("sample_rate", sample_rate))  { ROS_ERROR("'sample_rate' not set."); return 1; }
    pnh.param<std::string>("topic", topic, "/eeg_fbcsp");

    ROS_INFO("[SldaPublisher] Loading: %s", csv_filename.c_str());
    Eigen::MatrixXd full_data;
    try { full_data = readCSV<double>(csv_filename); }
    catch (const std::exception& e) { ROS_ERROR("CSV error: %s", e.what()); return 1; }

    int n_features = (int)full_data.cols();
    int total_rows = (int)full_data.rows();
    ROS_INFO("[SldaPublisher] Loaded: %d rows × %d features.", total_rows, n_features);

    int nbands;
    pnh.param("nbands", nbands, 5);
    int ncomponents = n_features / nbands;

    std::vector<double> default_bands = {8.0,10.0, 10.0,12.0, 12.0,14.0, 8.0,14.0, 14.0,20.0};
    std::vector<double> bands_d;
    if (!pnh.getParam("bands", bands_d)) bands_d = default_bands;
    std::vector<float> bands_f(bands_d.begin(), bands_d.end());

    ros::Publisher pub = nh.advertise<processing_bci::eeg_fbcsp>(topic, 1);
    ros::Rate loop_rate(sample_rate);

    ROS_INFO("[SldaPublisher] Waiting for subscriber on %s ...", topic.c_str());
    while (ros::ok() && pub.getNumSubscribers() == 0)
        ros::Duration(0.5).sleep();

    ROS_INFO("[SldaPublisher] Subscriber connected. Starting publication.");

    uint32_t seq = 0;
    for (int row = 0; ros::ok() && row < total_rows; row++) {
        Eigen::VectorXd row_data = full_data.row(row);

        processing_bci::eeg_fbcsp msg;
        msg.header.stamp       = ros::Time::now();
        msg.seq                = seq;
        msg.nbands             = (uint32_t)nbands;
        msg.ncomponents_for_band = (uint32_t)ncomponents;
        msg.bands              = bands_f;
        msg.data.assign(row_data.data(), row_data.data() + row_data.size());

        pub.publish(msg);
        ROS_INFO_THROTTLE(1.0, "[SldaPublisher] Published seq %u", seq);

        seq++;
        ros::spinOnce();
        loop_rate.sleep();
    }
    ROS_INFO("[SldaPublisher] Done.");
    return 0;
}