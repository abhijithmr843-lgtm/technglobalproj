# Step 4 - Simple dashboard to view attendance records
# A small Tkinter window that loads attendance.csv and shows it in a table.
# Has a refresh button and shows the total count for today.

import tkinter as tk
from tkinter import ttk
import csv
import os
from datetime import datetime

ATTENDANCE_FILE = 'attendance.csv'


def load_records():
    rows = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
    return rows


class Dashboard:
    def __init__(self, root):
        self.root = root
        root.title("Face Recognition Attendance - Dashboard")
        root.geometry("600x450")

        title = tk.Label(root, text="Attendance Records", font=("Arial", 18, "bold"))
        title.pack(pady=10)

        # table
        self.tree = ttk.Treeview(root, columns=("Name", "Date", "Time"), show="headings")
        self.tree.heading("Name", text="Name")
        self.tree.heading("Date", text="Date")
        self.tree.heading("Time", text="Time")
        self.tree.pack(fill="both", expand=True, padx=15, pady=5)

        # bottom bar with count + refresh
        bottom = tk.Frame(root)
        bottom.pack(fill="x", padx=15, pady=10)

        self.count_label = tk.Label(bottom, text="", font=("Arial", 11))
        self.count_label.pack(side="left")

        refresh_btn = tk.Button(bottom, text="Refresh", command=self.refresh)
        refresh_btn.pack(side="right")

        self.refresh()

    def refresh(self):
        # clear old rows
        for item in self.tree.get_children():
            self.tree.delete(item)

        rows = load_records()
        today = datetime.now().strftime("%Y-%m-%d")
        today_count = 0

        # skip header row if present
        for row in rows:
            if len(row) < 3 or row[0] == "Name":
                continue
            # show name in a nicer format (remove id prefix and underscores)
            display_name = row[0].split("_", 1)[-1].replace("_", " ")
            self.tree.insert("", "end", values=(display_name, row[1], row[2]))
            if row[1] == today:
                today_count += 1

        self.count_label.config(text=f"Present today ({today}): {today_count}")


if __name__ == '__main__':
    root = tk.Tk()
    app = Dashboard(root)
    root.mainloop()
