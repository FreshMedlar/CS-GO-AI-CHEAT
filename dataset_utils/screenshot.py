# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 09:49:54 2023

@author: stefa
"""
import pyscreenshot as ImageGrab
import schedule

from datetime import datetime
import time
import os
import keyboard

# Take a screenshot of the window
def take_screenshot():
    
    print("[+] TAKING SCREENSHOT")
    date = str(datetime.now()).replace(":","-") # For the image name we use the date in which the screenshot is made
    image_name = f"screenshot-{date}"
    screenshot = ImageGrab.grab()
    
    filepath = os.path.join(".", "dataset", "images" ,f"{image_name}.png")

    screenshot.save(filepath)
        
    print("[+] SCREENSHOT TAKEN")
    return filepath

# Take the screenshot every one second. If the 'q' key is pressed, the program exit
def main():
    
    schedule.every(1).seconds.do(take_screenshot)
    
    while True:
        if keyboard.is_pressed("q"):
            print("[+] END")
            break
        
        schedule.run_pending()

if __name__ == "__main__":
    main()