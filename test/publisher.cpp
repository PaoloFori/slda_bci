#include <ros/ros.h>
#include <eigen3/Eigen/Dense> 
#include <vector>
#include <string>
#include "processing_bci/utils.hpp" 
// Includi l'header generato dal tuo messaggio personalizzato
#include <processing_bci/eeg_fbcsp.h> 

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_publisher");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~"); 

    std::string topic = "/eeg_fbcsp";
    std::string csv_filename;
    double sample_rate;

    if (!private_nh.getParam("csv_file", csv_filename)) {
        ROS_ERROR("Parametro 'csv_file' non impostato!");
        return 1;
    }
    if (!private_nh.getParam("sample_rate", sample_rate)) {
        ROS_ERROR("Parametro 'sample_rate' non impostato!");
        return 1;
    }

    ROS_INFO("Loading dati da: %s", csv_filename.c_str());
    Eigen::MatrixXd full_data;
    try {
        full_data = readCSV<double>(csv_filename); 
    } catch (const std::exception& e) {
        ROS_ERROR("Errore durante la lettura del CSV: %s", e.what());
        return 1;
    }
    
    int n_features = full_data.cols(); // Numero di colonne nel CSV
    int total_rows = full_data.rows();
    ROS_INFO("Dati caricati: %d righe x %d colonne.", total_rows, n_features);

    // Cambiato il tipo di messaggio nel Publisher
    ros::Publisher pub = nh.advertise<processing_bci::eeg_fbcsp>(topic, 1);
    ros::Rate loop_rate(sample_rate);

    // Valori Hardcoded per il messaggio eeg_fbcsp
    uint32_t hardcoded_nbands = 5; 
    uint32_t hardcoded_ncomponents = n_features / hardcoded_nbands; 
    std::vector<float> hardcoded_bands = {8.0, 10.0, 10.0, 12.0, 12.0, 14.0, 8.0, 14.0, 14.0, 20.0};

    ROS_INFO("waiting for a subscriber...");
    while (ros::ok() && pub.getNumSubscribers() == 0) {
        ros::Duration(0.5).sleep(); 
    }

    ros::Duration(2.0).sleep();
    ROS_INFO("Start publication.");

    int current_row = 0;
    while (ros::ok()) {
        
        if (current_row >= total_rows) {
            ROS_INFO("File CSV ended.");
            break; 
        }

        // Estrazione della riga singola dal CSV
        Eigen::VectorXd row_data = full_data.row(current_row);

        // Creazione del messaggio eeg_fbcsp
        processing_bci::eeg_fbcsp msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = "eeg_frame";
        
        msg.seq = current_row;
        msg.nbands = hardcoded_nbands;
        msg.ncomponents_for_band = hardcoded_ncomponents;
        msg.bands = hardcoded_bands;

        // Conversione dati da Eigen a std::vector<float> per il messaggio
        std::vector<float> data_float(row_data.data(), row_data.data() + row_data.size());
        msg.data = data_float;

        pub.publish(msg);
        
        ROS_INFO("Sended sample seq: %d", msg.seq);

        current_row++;
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}