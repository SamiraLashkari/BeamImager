# Aquaculture Installation Volume Calculator

## Overview

This repository provides code for visualizing and calculating the volume of aquaculture installations (it tested for seaweed and mussels) using processed data from a multibeam ecosounder sensor.

## Data Requirements

The uploaded data should have the following six columns in this specific order: 'x', 'y', 'z', 'intensity', 'ping_number', 'beam' and be .txt or .csv format. Please ensure that there are no indices for the columns.

You can find sample data [here](https://drive.google.com/file/d/15rBiuT27kIJQX5pq0yL5JK0sN9E-XCw0/view?usp=drive_link) to process with the developed GUI.

## Requirements

- **Python 3.9.17**: Please install Python 3.9.17 in your environment, as the open3d library is not compatible with more recent versions of Python.
  
- **Dependencies**: Install all necessary Python packages by running the following command in the terminal of your environment:
  
```bash
pip install -r requirements.txt
```
## Running the GUI

To run the graphical user interface (GUI), execute the following command in your terminal:

```bash
python BeamImager_GUI_v2.py
```
![GUI-image](https://github.com/SamiraLashkari/BeamImager/blob/main/GUI_image.png)

