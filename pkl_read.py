import pickle
import pandas as pd

class CTPE:
    def __init__(self, name, is_positive, parser, num_slice, first_appear, avg_bbox, last_appear):
        self.study_num = name
        self.is_positive = is_positive
        self.phase = parser
        self.num_slice = num_slice
        self.first_appear = first_appear
        self.bbox = avg_bbox
        self.last_appear = last_appear

    def __len__(self):
        return self.num_slice

