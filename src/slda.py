#!/usr/bin/env python3

import rospy
from processing_bci.msg import eeg_fbcsp
from rosneuro_msgs.msg import NeuroOutput
import numpy as np
import yaml

# Importiamo l'oggetto di scikit-learn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Slda:
    def __init__(self):
        rospy.init_node('slda', anonymous=True)
        self.slda_name = "slda_model"
        
        try:
            self.path_decoder = rospy.get_param('~path_slda_model')
        except KeyError as e:
            rospy.logfatal(f"[{self.slda_name}] Mandatory parameter: 'path_slda_model'. Error:{e}.")
            return
            
        topic_sub = rospy.get_param('~topic_sub', '/eeg_fbcsp')
        
        try:
            self.slda_paradigm = rospy.get_param('~paradigm')
        except KeyError as e:
            rospy.logfatal(f"[{self.slda_name}] Mandatory parameter: 'paradigm'. Error:{e}.")
            return
        self.slda_name += f"_{self.slda_paradigm}"
        
        conf = self.configure()
        if not conf:
            rospy.logfatal(f"[{self.slda_name}] Error in the sLDA configuration.")
            return
        else:
            rospy.loginfo(f"[{self.slda_name}] sLDA configurated correctly.")


        rospy.Subscriber(topic_sub, eeg_fbcsp, self.callback)
        self.pub = rospy.Publisher(f'/{self.slda_paradigm}/neuroprediction/raw', NeuroOutput, queue_size=10)
        
        rospy.spin()

    
    def configure(self):
        try:
            with open(self.path_decoder, 'r') as file:
                yaml_data = yaml.safe_load(file)
                save_dict = yaml_data['sLDACfg']['params']
                
        except KeyError as e:
            rospy.logerr(f"[{self.slda_name}] Error missing key: {e}")
            return False
        except Exception as e:
            rospy.logerr(f"[{self.slda_name}] Error loading sLDA YAML file: {e}")
            return False
            
        try:
            self.clf = LinearDiscriminantAnalysis()
            self.clf.coef_ = np.array(save_dict['slda_weights'])
            self.clf.intercept_ = np.array(save_dict['slda_intercept'])
            self.clf.classes_ = np.array(save_dict['classes'])
            
            self.classes_ = self.clf.classes_
            self.nclasses = len(self.classes_)
            
            self.bands = save_dict['bands']
            
            self.nfeatures = len(save_dict['bands']) * len(save_dict['selected_components_indices'])
            
        except Exception as e:
            rospy.logerr(f"[{self.slda_name}] Error parsing the sLDA's parameter: {e}")
            return False

        return True
        
        
    def extract_features(self, msg):
        data = np.array(msg.data)
        
        if len(msg.bands) % 2 != 0:
            rospy.logwarn_throttle(1.0, f"[{self.slda_name}] Array bands wrong format (len {len(msg.bands)}).")
            return None
        
        msg_bands = [[msg.bands[i], msg.bands[i+1]] for i in range(0, len(msg.bands), 2)]
        n_comp = msg.ncomponents_for_band
        ordered_features = []
        
        for expected_band in self.bands:
            match_idx = -1
            for i, b in enumerate(msg_bands):
                if np.allclose(expected_band, b):
                    match_idx = i
                    break
                    
            if match_idx == -1:
                rospy.logwarn_throttle(1.0, f"[{self.slda_name}] Band {expected_band} asked for sLDA NOT present in the FBCSP message!")
                return None
                
            start_idx = match_idx * n_comp
            end_idx = start_idx + n_comp
            
            band_features = data[start_idx:end_idx]
            ordered_features.extend(band_features)
            
        ordered_features = np.array(ordered_features)
        
        if len(ordered_features) != self.nfeatures:
            rospy.logwarn_throttle(1.0, f"[{self.slda_name}] Expected {self.nfeatures} features, got {len(ordered_features)}.")
            return None

        dfet = np.log(ordered_features)
        
        return dfet


    def callback(self, msg):
        dfet = self.extract_features(msg)
        if dfet is None:
            return
            
        dfet = dfet.reshape(1, -1)
        probabilities = self.clf.predict_proba(dfet)[0]
        
        hard_pred_vector = np.zeros(self.nclasses, dtype=int)
        hard_pred_vector[np.argmax(probabilities)] = 1
        
        output = NeuroOutput()
        output.header.stamp = rospy.Time.now()
        output.neuroheader.seq = msg.seq
        output.softpredict.data = probabilities.tolist()
        output.hardpredict.data = hard_pred_vector.tolist() 
        output.decoder.type = self.slda_name
        output.decoder.path = self.path_decoder
        output.decoder.classes = self.classes_.astype(int).tolist()
        
        self.pub.publish(output)

if __name__ == '__main__':
    Slda()