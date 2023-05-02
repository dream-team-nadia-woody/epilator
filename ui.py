import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from typing import Any, List, Tuple, Union
from typing_extensions import Literal
from video.vid import Video
from os import path
from video.conversion import Conversions
from PIL import ImageTk
import json

class FlaggerUI(tk.Frame):
    filename: str
    vid: Video
    frame_no: tk.IntVar
    img: ImageTk.PhotoImage
    current_interval: Tuple[int, int] = (-1, -1)
    intervals: List[Tuple[int, int]] = []

    def __init__(self, root: tk.Tk, filename: Union[str, None] = None) -> None:
        super().__init__(root)
        if filename is None:
            self.filename = filedialog.askopenfilename(defaultextension='.mp4')
        self.filename = filename
        self.vid = Video.from_file(filename, resize=(200, 200))
        self.frame_no = tk.IntVar(self)
        root.eval('tk::PlaceWindow . center')
        slider = ttk.Scale(self, from_=0, to=self.vid.frame_count-1,
                           variable=self.frame_no, command=self.update_frame, orient='horizontal')
        self.img = ImageTk.PhotoImage(self.vid[0].show(2.0))
        img_label = ttk.Label(self, image=self.img, name="frame")
        img_label['image'] = self.img
        nudge_up = tk.Button(
            self, text=">>", name="nudge_up", command=self.nudge_frame_up, foreground='black')
        nudge_down = tk.Button(
            self, text="<<", name="nudge_down", command=self.nudge_frame_down, foreground='black')
        set_frame = tk.Frame(self)
        set_start = tk.Button(set_frame, background="#BBFFBB",
                              foreground='black', text="Set Start", command=self.set_start)
        set_end = tk.Button(set_frame, background="#FFBBBB",
                            foreground='black', text="Set End", command=self.set_start)
        set_start.grid(row=0, column=0, sticky="EW")
        set_end.grid(row=0, column=1, sticky="EW")
        nudge_down.grid(row=0, column=0, sticky="NSE")
        img_label.grid(row=0, column=1)
        nudge_up.grid(row=0, column=2, sticky="NSW")
        slider.grid(row=1, column=0, columnspan=3, sticky="EW")
        set_frame.grid(row=2, column=1)

    def update_frame(self, event):
        self.img = ImageTk.PhotoImage(self.vid[self.frame_no.get()].show(2.0))
        self.children['frame']['image'] = self.img

    def nudge_frame_up(self):
        frame_no = self.frame_no.get()
        if frame_no < self.vid.frame_count:
            self.frame_no.set(frame_no + 1)
            self.update_frame(None)

    def nudge_frame_down(self):
        frame_no = self.frame_no.get()
        if frame_no > 0:
            self.frame_no.set(frame_no - 1)
            self.update_frame(None)

    def set_start(self):
        start = self.frame_no.get()
        if self.current_interval[1] > -1:
            self.intervals.append((start, self.current_interval[1]))
            self.current_interval = (-1, -1)
        else:
            self.current_interval = (start, -1)

    def set_end(self):
        end = self.frame_no.get()
        if self.current_interval[0] > -1:
            print("saving interval")
            # self.intervals.append((self.current_interval[0], end))
            self.current_interval = (-1, -1)
        else:
            self.current_interval = (-1, end)


if __name__ == "__main__":
    root = tk.Tk()
    frame = FlaggerUI(root, 'videos/Banned Pokemon Seizure Scene.mp4')
    frame.pack()
    root.mainloop()
    intervals = [{"start":start,"end":end} for start,end in frame.intervals]
    with open(path.splitext(frame.filename)[0] + '.json', 'w+') as f:
        json.dump(intervals,f)
