import cv2
import pickle
import tkinter as tk
from tkinter import filedialog, PanedWindow, simpledialog, Listbox, Scrollbar
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import re

LABEL_FILE_NAME = "label.pkl"
VIDEO_NAME = "sample_left.mp4"

LEVEL_NAME = ["skills", "done"]
ITEM_LABEL_NAME = ["", "can", "cup", "plate", "tissue", "sponge"]
LABEL_NAME = [range(10), ] + [range(10)]
NUM_LEVELS = len(LEVEL_NAME)


class ProgressBarOne(tk.Canvas):
    def __init__(self, master, colors, update_callback, total_frames=100, width=300, height=30, **kwargs):
        super().__init__(master, width=width, height=height, **kwargs)
        self.progress = 0 
        self.dragging = False 
        self.update_callback = update_callback
        self.labels = [0] * total_frames
        self.colors = colors
        self.total_frames = total_frames
        self.last_draw = -1

        self.range = [0, self.total_frames]
        # self.scrollbar = tk.Scrollbar(self.master, orient=tk.VERTICAL, command=self.yview)
        # self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scrollbar = tk.Scale(
            self.master,
            from_=1, 
            to=100,  
            orient="horizontal",  
            command=self.on_scroll,  
        )
        self.scrollbar.set(100)  
        self.scrollbar.pack(fill=tk.X, pady=10)
        self.scrollbar_start = tk.Scale(
            self.master,
            from_=0,  
            to=self.total_frames, 
            orient="horizontal",  
            command=self.on_scroll_start, 
        )
        self.scrollbar_start.set(0) 
        self.scrollbar_start.pack(fill=tk.X, pady=10)
        # self.configure(yscrollcommand=self.scrollbar.set)
        # self.bind_all("<MouseWheel>", self.on_mousewheel)


        self.bind("<Button-1>", self.toggle_dragging) 
        self.bind("<Motion>", self.mouse_move)
        self.focus_set()
        self.bind("<KeyPress-Left>", self.key_left) 
        self.bind("<KeyPress-Right>", self.key_right) 
        # listen the scroll of mouse middle
        self.bind()

        self.draw_progress()

    def on_scroll(self, value):
        # print("self.", self.total_frames, value, type(value))
        # self.range = [0, int(self.total_frames * int(value)/100)]
        l = int(self.total_frames * int(value)/100)
        self.range[1] = min(self.range[0] + l, self.total_frames)
        self.range[0] = min(self.range[1]-l, self.range[0])
        
        self.scrollbar_start.config(from_=0, to=self.total_frames-self.c_total_frames)
        self.draw_progress()

    def on_scroll_start(self, value):
        # print("self.", self.total_frames, value, type(value))
        s = int(value)
        if s + self.c_total_frames > self.total_frames:
            self.scrollbar_start.set(self.total_frames - self.c_total_frames)
            return
        l = self.c_total_frames
        self.range[0] = s
        self.range[1] = s+l
        self.draw_progress()

    def reset(self, labels, total_frames):
        self.progress = 0
        self.labels = labels
        self.total_frames = total_frames
        self.last_draw = -1
        self.scrollbar.set(100)
        self.scrollbar_start.set(0)
        self.scrollbar_start.config(from_=0, to=total_frames)
        self.range = [0, self.total_frames]
        self.draw_progress()

    def toggle_dragging(self, event):
        self.dragging = not self.dragging
        if self.dragging:
            self.mouse_move(event)

    @property
    def c_total_frames(self):
        return self.range[1] - self.range[0]

    def mouse_move(self, event):
        if self.dragging:
            x = event.x
            width = self.winfo_width()

            self.progress = max(0, min(self.c_total_frames, int(x / width * self.c_total_frames))) + self.range[0]
            self.draw_progress()
            self.update_callback(self.progress)

    def key_left(self, event):
        self.progress = max(0, self.progress - 1)
        self.draw_progress()

    def key_right(self, event):
        self.progress = min(1.0, self.progress + 1)
        self.draw_progress()

    def draw_progress(self):
        self.update_classification_bar()

    
    def update_classification_bar(self, labels=None, current_frame=None, force=True):
        if labels is not None:
            self.labels = labels
        if current_frame is not None:
            self.progress = current_frame

        if not force and self.last_draw == self.progress:
            return
        self.last_draw = self.progress

        self.delete("all")
        # print("self", self.range, self.c_total_frames)
        width = self.winfo_width()
        height = self.winfo_height()
        box_width = width / self.c_total_frames
        for i in range(self.c_total_frames):
            color = self.colors.get(self.labels[i+self.range[0]], "gray")
            self.create_rectangle(i * box_width, 0, (i + 1) * box_width, height, fill=color, outline="")
        
        for i in range(self.c_total_frames):
            if (i+self.range[0]) % 30 == 0:
                self.create_rectangle(i * box_width, 0, (i + 1) * box_width, height, fill="gray", outline="")
        i = self.progress - self.range[0]
        color = "white"
        gap = 5
        # self.create_rectangle(i * box_width, 0+gap, (i + 1) * box_width, height-gap, fill=color, outline="")
        self.create_rectangle(i * box_width, 0+gap, (i + 1) * box_width, height-gap, fill=color, outline="")

class ProgressBar(tk.Canvas):
    def __init__(self, master, colors, update_callback, num_levels=1, total_frames=100, width=300, height=30, **kwargs):
        super().__init__(master, width=width, height=height, **kwargs)
        self.progress = 0
        self.dragging = False
        self.update_callback = update_callback
        self.labels = [[0]*num_levels] * total_frames
        self.colors = colors
        self.total_frames = total_frames
        self.last_draw = -1

        self.range = [0, self.total_frames]
        # self.scrollbar = tk.Scrollbar(self.master, orient=tk.VERTICAL, command=self.yview)
        # self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scrollbar = tk.Scale(
            self.master,
            from_=1,
            to=100,
            orient="horizontal",
            command=self.on_scroll,
        )
        self.scrollbar.set(100)
        self.scrollbar.pack(fill=tk.X, pady=10)
        self.scrollbar_start = tk.Scale(
            self.master,
            from_=0,
            to=self.total_frames, 
            orient="horizontal", 
            command=self.on_scroll_start, 
        )
        self.scrollbar_start.set(0)
        self.scrollbar_start.pack(fill=tk.X, pady=10)

        self.num_levels = num_levels
        self.real_levels = num_levels
        self.level_focus = 0

        self.bind("<Button-1>", self.toggle_dragging)
        self.bind("<Motion>", self.mouse_move)
        self.focus_set()
        self.bind("<KeyPress-Left>", self.key_left)
        self.bind("<KeyPress-Right>", self.key_right)
        # w
        self.bind("<KeyPress-q>", self.change_level)
        # s
        self.bind("<KeyPress-e>", self.change_level)
        self.draw_progress()

    def on_scroll(self, value):
        # print("self.", self.total_frames, value, type(value))
        # self.range = [0, int(self.total_frames * int(value)/100)]
        l = int(self.total_frames * int(value)/100)
        self.range[1] = min(self.range[0] + l, self.total_frames)
        self.range[0] = min(self.range[1]-l, self.range[0])
        
        self.scrollbar_start.config(from_=0, to=self.total_frames-self.c_total_frames)
        self.draw_progress()

    def on_scroll_start(self, value):
        # print("self.", self.total_frames, value, type(value))
        s = int(value)
        if s + self.c_total_frames > self.total_frames:
            self.scrollbar_start.set(self.total_frames - self.c_total_frames)
            return
        l = self.c_total_frames
        self.range[0] = s
        self.range[1] = s+l
        self.draw_progress()
    
    def reset(self, labels, total_frames, real_levels):
        self.progress = 0
        self.labels = labels
        self.total_frames = total_frames
        self.last_draw = -1
        self.scrollbar.set(100)
        self.scrollbar_start.set(0)
        self.scrollbar_start.config(from_=0, to=total_frames)
        self.range = [0, self.total_frames]
        self.real_levels = real_levels
        self.draw_progress()

    @property
    def c_total_frames(self):
        return self.range[1] - self.range[0]

    def change_level(self, event):
        if event.keysym == "q":
            self.level_focus = (self.level_focus - 1) % self.num_levels
        elif event.keysym == "e":
            self.level_focus = (self.level_focus + 1) % self.num_levels
        # self.update_classification_bar()
        self.draw_progress()
        print("change level", self.level_focus)

    def toggle_dragging(self, event):
        self.focus_set()
        self.dragging = not self.dragging 
        if self.dragging:
            self.mouse_move(event)

    def mouse_move(self, event):
        if self.dragging:
            x = event.x
            width = self.winfo_width()
            
            self.progress = max(0, min(self.c_total_frames, int(x / width * self.c_total_frames))) + self.range[0]
            self.draw_progress()
            self.update_callback(self.progress)

    def key_left(self, event):
        self.progress = max(0, self.progress - 1)
        self.draw_progress()

    def key_right(self, event):
        self.progress = min(1.0, self.progress + 1)
        self.draw_progress()

    def draw_progress(self):
        self.update_classification_bar(real_levels=self.real_levels)
    
    def update_classification_bar(self, labels=None, current_frame=None, force=True, real_levels=None):
        if labels is not None:
            self.labels = labels
        if current_frame is not None:
            self.progress = current_frame

        if not force and self.last_draw == self.progress:
            return

        self.last_draw = self.progress

        self.delete("all")
        width = self.winfo_width()
        height = self.winfo_height()
        box_width = width / self.c_total_frames
        height_ = height / self.num_levels
        s_height = 0

        gap = 5
        i = self.progress
        color = "white"

        for li in range(self.num_levels):
            color_start = -1
            last_colot = -1
            for i in range(self.c_total_frames):
                color = self.colors.get(self.labels[i+self.range[0]][li], "gray")
                if color != last_colot or i == self.c_total_frames-1:
                    j = color_start
                    self.create_text(j * box_width, s_height+height_/2, text=LABEL_NAME[li][int(self.labels[i+self.range[0]-1][li])], fill="white", anchor="w")
                    color_start = i
                    last_colot = color
                self.create_rectangle(i * box_width, s_height+gap//2, (i + 1) * box_width, s_height + height_-gap//2, fill=color, outline="")

            for i in range(self.c_total_frames):
                if (i+self.range[0]) % 30 == 0:
                    self.create_rectangle(i * box_width, s_height+gap//2, (i + 1) * box_width, s_height + height_-gap//2, fill="gray", outline="")

            level_name = LEVEL_NAME[li]
            # if real_levels is not None and li >= real_levels:
            if li >= real_levels:
                # level_name = upper(level_name)
                # uppercase
                level_name = level_name.upper()
                level_name += "(?)"
            self.create_text(10, s_height+height_/2, text=level_name, fill="white", anchor="w")

            if self.level_focus == li:
                j = self.progress - self.range[0]
                color = "white"
                self.create_rectangle(j * box_width, s_height+gap, (j + 1) * box_width, s_height+height_-gap, fill=color, outline="")
            s_height += height_
        
            # self.create_rectangle(i * box_width, 0+gap, (i + 1) * box_width, height-gap, fill=color, outline="")
        # self.create_rectangle(i * box_width, 0+gap, (i + 1) * box_width, height-gap, fill=color, outline="")


class VideoLabeler:
    def __init__(self, root, num_levels = NUM_LEVELS):
        self.root = root
        self.root.title("Video Labeler")

        # Load video and data
        self.video_path = "video.mp4"
        self.data_path = "label.pkl"
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.num_levels = num_levels
        self.real_levels = num_levels
        self.labels = self.load_labels(self.data_path)
        self.current_frame = 0
        self.start_id = -1
        self.colors = {0: "red", 1: "green", 2: "blue", 3: "#aaaa00", 
                       4: "#00aaaa", 5: "orange", 6: "pink", 7: "cyan", 8: "magenta", 9: "lime"}
        self.playing = False
        self.play_cls = -1
        
        # load dir
        self.directory = "place_can_R"
        self.current_directory = None
        self.video_files = []
        self.current_file = None
        self.modified = False
        
        # Set up the GUI
        self.setup_ui()

        # Bind key presses for labeling
        self.root.bind("<Key>", self.label_frame)
        self.root.bind("<Control-s>", self.save_labels)
        # bind ctrl-w as close window
        self.root.bind("<Control-w>", self.try_to_close)

    def check_save(self):
        if self.modified:
            ret = messagebox.askyesnocancel("Save", "Do you want to save the labels?")
            print("save ask return", ret)
            if ret is None:
                return False
            elif ret:
                self.save_labels()
            return True
        return True

    def try_to_close(self, event):
        # self.save_labels()
        if not self.check_save():
            return
        self.root.destroy()

    def setup_ui(self):
        # self.cap = cv2.VideoCapture(self.video_path)
        # self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Paned window for adjustable list and viewer
        self.paned_window = PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Folder input and selection
        self.input_frame = tk.Frame(self.paned_window)
        self.paned_window.add(self.input_frame, width=200)

        # Load directory button
        self.browse_button = tk.Button(self.input_frame, text="Browse", command=self.load_folder)
        self.browse_button.pack(fill=tk.X)
        # Scrollable list of videos
        self.scrollbar = Scrollbar(self.input_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = Listbox(self.input_frame, yscrollcommand=self.scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.listbox.yview)
        self.listbox.bind('<<ListboxSelect>>', self.change_video)

        # Main viewer
        self.viewer_frame = tk.Frame(self.paned_window)
        self.paned_window.add(self.viewer_frame, stretch="always")

        # Display current file path
        self.file_label = tk.Label(self.viewer_frame, text="", bg="white", anchor="w")
        self.file_label.pack(fill=tk.X)

        # self.file_label = tk.Text(self.viewer_frame, height=1, borderwidth=0)
        # self.file_label.insert(1.0, "Hello, world!")
        # self.file_label.pack(fill=tk.X)

        # Image display
        self.canvas = tk.Canvas(self.viewer_frame, width=self.cap.get(3), height=self.cap.get(4))
        self.canvas.pack()
        
        # Classification bar display
        # self.bar_canvas = tk.Canvas(self.viewer_frame, width=500, height=50)
        # self.bar_canvas.pack(fill="x", expand=True)
        if self.num_levels == 1:
            self.progress_bar = ProgressBarOne(self.viewer_frame, self.colors, self.update_frame, width=500, height=100)
        else:
            self.progress_bar = ProgressBar(self.viewer_frame, self.colors, self.update_frame, width=500, height=100, num_levels=self.num_levels)
        self.progress_bar.pack(fill="x", expand=True)

        # Slider for frame navigation
        self.slider = tk.Scale(self.viewer_frame, from_=0, to=self.total_frames-1, orient="horizontal", command=self.slider_used)
        self.slider.pack(fill="x", expand=True)

        # Button to save labels
        self.save_button = tk.Button(self.viewer_frame, text="Save Labels", command=self.save_labels)
        self.save_button.pack()

        # Button to batch label
        self.batch_label_button = tk.Button(self.viewer_frame, text="Load logs", command=self.batch_label)
        self.batch_label_button.pack()

        self.text_button = tk.Button(self.viewer_frame, text="Change Label", command=self.change_label)
        self.text_button.pack()

        # Display the first frame
        self.load_folder(self.directory)
        self.update_frame(self.current_frame)
        self.update_classification_bar()

    def change_label(self,):
        text = simpledialog.askstring("Input", "Change all label a b to c d:\nExample:a b,c d", initialvalue="1 3,3 2", parent=self.root)
        if text is not None:

            print("change label:", text)
            old_label, new_label = text.split(",")
            old_label = [int(i) for i in old_label.strip().split(" ")]
            new_label = [int(i) for i in new_label.strip().split(" ")]
            clone_label = self.labels.copy()
            for i, a, b in zip(range(len(old_label)), old_label, new_label):
                clone_label[self.labels == a] = b
            
            self.labels = clone_label

            self.update_classification_bar()
        # self.text_button.config(text="Get Path")
        # self.text_button.pack()
        # self.text_button.pack_forget()
        # self.text_button.destroy()

    def get_text(self,):
        print("get text")
        text = simpledialog.askstring("Input", "File logs:", show=self.current_directory, parent=self.root)
        # self.text_button.config(text="Get Path")
        # self.text_button.pack()
        # self.text_button.pack_forget()
        # self.text_button.destroy()
    
    def load_folder(self, directory=None):
        if directory is None:
            directory = filedialog.askdirectory()
        if directory:
            self.directory = directory
            self.video_files = []
            # recursively search for video files, "xxx/video/left_sample.mp4, show as xxx/"
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('left.mp4'):
                        self.video_files.append(os.path.relpath(os.path.join(root, file), directory))
            
            self.video_files.sort()
            # self.video_files = [file for file in os.listdir(directory) if file.endswith('.mp4')]
            self.listbox.delete(0, tk.END)
            for file in self.video_files:
                # check if data file exists
                print("data_file", directory, file, LABEL_FILE_NAME)
                data_file = os.path.join(directory, os.path.dirname(file), LABEL_FILE_NAME)
                print("data_file", data_file)
                if os.path.exists(data_file):
                    file = f"*{file}"
                else:
                    file = f" {file}"
                self.listbox.insert(tk.END, file)
                
            # default load first video
            if not self.check_save():
                return
            self.current_file = self.video_files[0]
            self.load_video()

    def slider_used(self, event):
        pass
        # self.current_frame = int(self.slider.get())
        # self.update_frame(self.current_frame)

    def update_frame(self, frame_number):
        self.current_frame = frame_number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lb = self.labels[frame_number]
            cv2.putText(frame, f"Frame {frame_number}, CLS {lb}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            img = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.update_classification_bar()

    @property
    def level_focus(self):
        return self.progress_bar.level_focus
    
    def label_frame(self, event):
        if event.char.isdigit():
            # self.labels[self.current_frame] = int(event.char)
            # self.update_classification_bar()
            
            # print(f"Frame {self.current_frame} labeled as {event.char}")
            
                
            if self.playing:
                self.play_cls = int(event.char)
            else:
                if self.start_id == -1:
                    self.start_id = self.current_frame
                else:
                    print(f"Frames {self.start_id} to {self.current_frame} labeled as {event.char}")
                    self.modified = True
                    if self.current_frame < self.start_id:
                        self.start_id, self.current_frame = self.current_frame, self.start_id
                    for frame in range(self.start_id, self.current_frame + 1):
                        if self.num_levels == 1:
                            self.labels[frame] = int(event.char)
                        else:
                            self.labels[frame, self.level_focus] = int(event.char)
                    self.update_classification_bar()
                    self.start_id = -1
            self.update_frame(self.current_frame)
        # left arrow
        elif event.keysym == "Left":
            self.current_frame = max(0, self.current_frame - 1)
            self.update_frame(self.current_frame)
        # right arrow
        elif event.keysym == "Right":
            self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
            self.update_frame(self.current_frame)

        elif event.keysym == "space":
            self.playing = not self.playing
            if self.playing:
                self.play_cls = self.labels[self.current_frame]
                self.play_video()
        # ctrl+s to save labels
        elif event.keysym == "s" and event.state == 4:
            # self.save_labels()
            print("haha")

    def update_classification_bar(self):
        if self.num_levels == 1:
            self.progress_bar.update_classification_bar(self.labels, self.current_frame)
        else:
            self.progress_bar.update_classification_bar(self.labels, self.current_frame, real_levels=self.real_levels)
        # self.bar_canvas.delete("all")
        # box_width = 800 / self.total_frames
        # for i in range(self.total_frames):
        #     color = self.colors.get(self.labels[i], "gray")
        #     self.bar_canvas.create_rectangle(i * box_width, 0, (i + 1) * box_width, 50, fill=color, outline="")
        # i = self.current_frame
        # color = "white"
        # gap = 5
        # self.bar_canvas.create_rectangle(i * box_width, 0+gap, (i + 1) * box_width, 50-gap, fill=color, outline="")
        
    def change_video(self, event):
        selection = event.widget.curselection()
        if selection:
            if not self.check_save():
                return
            index = selection[0]
            self.current_file = self.video_files[index]
            self.load_video()
            
    def load_video(self, path=None):
        if path is None:
            if self.current_file:
                print("load file", os.path.join(self.directory, self.current_file))
                path = os.path.join(self.directory, self.current_file)

        if path is None:
            return
        
        self.file_label.config(text=path)

        self.modified = False

        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.canvas.config(width=self.cap.get(3), height=self.cap.get(4))
        
        self.slider.config(to=self.total_frames-1)
        sub_dir = os.path.dirname(path)
        self.current_directory = sub_dir
        self.labels = self.load_labels(os.path.join(sub_dir, LABEL_FILE_NAME))
        print("Load labels", self.labels.shape, self.total_frames)

        # self.progress_bar.update_classification_bar(labels=self.labels, total_frames=self.total_frames-1)
        if self.num_levels == 1:
            self.progress_bar.reset(labels=self.labels, total_frames=self.total_frames-1)
        else:
            self.progress_bar.reset(labels=self.labels, total_frames=self.total_frames-1, real_levels=self.real_levels)
        self.update_frame(0)

    def init_done_label(self, skill_label):
        last_lbl = 0
        last_left = len(skill_label) + 1
        new_lbl = np.zeros_like(skill_label)
        for i in range(len(skill_label)):
            if last_lbl == 0 and skill_label[i] != 0:
                last_left = i
                last_lbl = skill_label[i]
            elif last_lbl != 0 and (i+1 >= len(skill_label) or skill_label[i+1] != last_lbl):
                new_lbl[(i+last_left)//2: i] = 1
                last_lbl = 0
        return new_lbl


    def load_labels(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                labels =  pickle.load(f)
            self.real_levels = 1
            if len(labels) != self.total_frames:
                print("[WARNING] label number less than total frames", len(labels), self.total_frames)
                if len(labels) +1 == self.total_frames:
                    print("[INFO] quick fix: only add the last frame")
                    labels = np.concatenate([labels, np.zeros(1)])

            if self.num_levels > 1:
                if len(labels.shape) == 1:
                    print("[WARNING] label shape less than num_levels", labels.shape, self.num_levels)
                    labels = labels[:, np.newaxis]
                self.real_levels = labels.shape[1]
                if len(labels.shape) == 2 and labels.shape[1] == self.num_levels-1:
                    print("[WARNING] label shape less than num_levels", labels.shape, self.num_levels)
                    # new_lbl = labels[:, :1] !=0
                    new_lbl = self.init_done_label(labels[:, :1])
                    labels = np.concatenate([labels, new_lbl], axis=1)
                    
            return labels
        except:
            if self.num_levels == 1:
                return np.zeros(self.total_frames, dtype=np.uint8)
            else:
                return np.zeros((self.total_frames, self.num_levels), dtype=np.uint)

    def play_video(self):
        if self.playing and self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.update_classification_bar()
            self.update_frame(self.current_frame)
            self.modified = True
            self.labels[self.current_frame] = self.play_cls
            self.root.after(2, self.play_video)  # 1000 ms / 60 fps ≈ 17 ms per frame

    def save_labels(self, event=None):
        if self.current_directory:
            self.modified = False
            with open(os.path.join(self.current_directory, LABEL_FILE_NAME), 'wb') as f:
                pickle.dump(self.labels, f)
            print("Labels saved.")

    # def save_labels(self, event=None):
    #     with open(self.data_path, 'wb') as f:
    #         pickle.dump(self.labels, f)
    #     print("Labels saved.")

    def batch_label(self):
        # start = simpledialog.askinteger("Input", "Start frame:", parent=self.root)
        # end = simpledialog.askinteger("Input", "End frame:", parent=self.root)
        # label = simpledialog.askinteger("Input", "Label:", parent=self.root)
        file_logs = simpledialog.askstring("Input", "File logs:", parent=self.root)
        if file_logs is None:
            return
        if not self.check_save():
            return
        
        for line in file_logs.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("Frames"):
                #Frames 1142 to 1221 labeled as 2
                start, end, label = map(int, re.findall(r"\d+", line))
                self.modified = True
                for frame in range(start, end + 1):
                    self.labels[frame] = label
            elif line.startswith("load file"):
                fpath = line.split(" ")[-1]
                self.load_video(fpath)
        # if start is not None and end is not None and label is not None:
        #     for frame in range(start, end + 1):
        #         self.labels[frame] = label


        self.update_classification_bar()
            # print(f"Frames {start} to {end} labeled as {label}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoLabeler(root)
    root.mainloop()
