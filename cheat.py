# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 19:55:34 2023

@author: stefa
"""
    
# Main script. From here the analyzer is created,
# then the window name is passed through a method
# and finally the weights are passed.
#
# After all the initialization the instance enters
# a loop in which every frame from the game is analyzed
from video_analyzer import VideoAnalyzer

if __name__ == '__main__':
    
    analyzer = VideoAnalyzer()
    analyzer.get_window("counter-strike")
    analyzer.init_model("./object detection/model_best.pt")
    
    analyzer.analyze_video_feed()
    
    