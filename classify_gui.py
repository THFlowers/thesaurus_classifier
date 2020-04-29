#!/usr/bin/env python3

# Standard library
import csv
import json
import gc
from threading import Thread
from queue import Queue
from os import path
from tkinter import *
from tkinter import messagebox

# 3rd party
import wordsegment

# Our code
from make_model import generate_frequencies
from classify import classify
from classify_test import valid_models, TFidF

DEFAULT_FILE = "labeledData.csv"


# Template for long running async task from: http://zetcode.com/articles/tkinterlongruntask/
# Though that used Process which proved useless with our code as SQLite3 and pickle don't agree at all
# Switched to oldschool python thread version, we want to share address space anyway
class Gui(Frame):

    def __init__(self, parent, queue):
        Frame.__init__(self, parent)

        self.parent = parent
        self.queue = queue
        self.resource_function = None
        self.stemmed_database = True
        self.my_idf = None

        # Limited use members, declaration here to suppress warnings
        self.thread = None
        self.wait_message = None

        self.frame_a = Frame()

        self.ds_label = Label(master=self.frame_a, text="Select Data Source")
        self.ds_label.grid(row=0, column=0)

        self.entry = Entry(master=self.frame_a)
        self.entry.insert(0, DEFAULT_FILE)
        self.entry.grid(row=0, column=1)

        self.ds_button = Button(master=self.frame_a, text="Make Model", command=self.make_model_callback)
        self.ds_button.grid(row=0, column=2)

        self.load_model_button = Button(master=self.frame_a, text="Load Model", command=self.load_model_callback)
        self.load_model_button.grid(row=0, column=3)

        self.frame_a.pack()

        self.frame_b = Frame()

        self.tb_label = Label(master=self.frame_b, text="Text to Classify")
        self.tb_label.pack(side=TOP, anchor=W)

        self.textbox = Text(master=self.frame_b)
        self.textbox.pack()

        self.frame_b.pack(side=LEFT, anchor=N)

        self.frame_c = Frame()

        self.lst_label = Label(master=self.frame_c, text="Configuration")
        self.lst_label.pack(side=TOP, anchor=W)

        self.model_list = Listbox(master=self.frame_c, selectmode=SINGLE)
        self.model_list.pack()

        for model in valid_models.keys():
            self.model_list.insert(END, model)

        self.model_list.select_set(END)

        self.model_list.bind('<<ListboxSelect>>', self.load_resource_callback)

        self.seg_var = BooleanVar()
        self.seg_button = Checkbutton(master=self.frame_c, text="Segment Words", variable=self.seg_var)
        self.seg_button.pack()

        self.num_sim_label = Label(master=self.frame_c, text="Num Similar")
        self.num_sim_label.pack()

        self.num_sim_box = Spinbox(master=self.frame_c, from_=0, to_=10)
        self.num_sim_box.delete(0, "end")
        self.num_sim_box.insert(0, 3)
        self.num_sim_box.pack()

        self.min_sim_label = Label(master=self.frame_c, text="Min Similarity %")
        self.min_sim_label.pack()

        self.min_sim_box = Spinbox(master=self.frame_c, from_=0, to_=100)
        self.min_sim_box.delete(0, "end")
        self.min_sim_box.insert(0, 20)
        self.min_sim_box.pack()

        self.run_button = Button(master=self.frame_c, text="Classify", command=self.classify_callback)
        self.run_button.configure(state='disable')
        self.run_button.pack(pady=20)

        self.class_label = Label(master=self.frame_c, text="No Class Chosen", fg='red')
        self.class_label.pack()

        self.frame_c.pack(side=RIGHT, anchor=N)

    def load_resource_callback(self, _event):

        self.class_label.configure(text="No Class Chosen")

        # If old resource exists, delete it
        if self.resource_function is not None:
            self.resource_function = None
            gc.collect()

        # Grab the name from the text stored in the model_list, via curselection (index)
        name = self.model_list.get(self.model_list.curselection()[0])
        if name == ():
            print("Odd selection event happened again")
            name = 'Raw'
            # TODO: Make model_list select 'Raw'

        # Grab stemmed_database value, which is 1st item in valid_models dictionary
        self.stemmed_database = valid_models[name][1]

        # Raw requires no work
        if name == 'Raw':
            self.model_list.configure(state='normal')
        else:
            # Create new toplevel window for loading message, make it take control and be unclosable until done
            self.wait_message = Toplevel(master=self.parent)
            self.wait_message.protocol("WM_DELETE_WINDOW", (lambda: None))
            self.wait_message.grab_set()

            width = self.parent.winfo_width() / 2
            height = self.parent.winfo_height() / 2
            self.wait_message.geometry("+%d+%d" % (width, height))

            wait_label = Label(master=self.wait_message, text="Loading")
            wait_label.pack()

            self.thread = Thread(target=(lambda que: que.put(valid_models[name][0]())), args=(self.queue,))
            self.thread.start()
            self.after(20, self.loading_loop)

    def loading_loop(self):
        if self.thread.is_alive():
            self.after(20, self.loading_loop)
        else:
            self.model_list.configure(state='normal')
            self.resource_function = q.get()
            self.wait_message.grab_release()
            self.wait_message.destroy()

    def load_model_callback(self):
        if not path.exists("doc_frequencies.json") or not path.exists("term_frequencies.json"):
            self.run_button.configure(state='disable')
            messagebox.showerror("doc/term Frequency Files Missing")
        else:
            self.run_button.configure(state='active')

            with open('term_frequencies.json', 'r') as fi:
                term_frequencies = json.load(fi)

            with open('doc_frequencies.json', 'r') as fi:
                doc_frequencies = json.load(fi)

            self.my_idf = TFidF(term_frequencies, doc_frequencies)

    def make_model_callback(self):
        valid = path.isfile(self.entry.get())
        if not valid:
            self.run_button.configure(state='disable')
        else:
            if self.my_idf is not None:
                self.my_idf = None
                gc.collect()

            self.wait_message = Toplevel(master=self.parent)
            self.wait_message.protocol("WM_DELETE_WINDOW", (lambda: None))
            self.wait_message.grab_set()

            width = self.parent.winfo_width() / 2
            height = self.parent.winfo_height() / 2
            self.wait_message.geometry("+%d+%d" % (width, height))

            wait_label = Label(master=self.wait_message, text="Loading")
            wait_label.pack()

            self.thread = Thread(target=self.make_model, args=(q,))
            self.thread.start()
            self.after(20, self.model_loop)

    def make_model(self, que):
        with open(self.entry.get(), 'r') as fi:
            labeled_data = csv.DictReader(fi, delimiter=',')

            (term_frequencies, doc_frequencies) = generate_frequencies(labeled_data)
            que.put(TFidF(term_frequencies, doc_frequencies))

    def model_loop(self):
        if self.thread.is_alive():
            self.after(20, self.model_loop)
        else:
            self.run_button.configure(state='active')
            self.my_idf = q.get()
            self.wait_message.grab_release()
            self.wait_message.destroy()

    def classify_callback(self):
        category = classify(self.my_idf,
                            self.textbox.get("1.0", "end-1c"),
                            sim_func=self.resource_function,
                            num_similar=int(self.num_sim_box.get()),
                            min_similarity=0.01 * int(self.min_sim_box.get()),
                            stemmed_database=self.stemmed_database,
                            segment=bool(self.seg_var.get()))

        self.class_label.configure(text=str(category[0]))


if __name__ == '__main__':
    wordsegment.load()

    window = Tk()
    q = Queue()
    app = Gui(window, q)
    window.mainloop()
