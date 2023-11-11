import pickle
test_file = '..\data\preprocessed_DIP_IMU_v1\dipimu_s_03_01.pkl'
data = pickle.load(open(test_file, "rb"))
imu = data['imu']


k=3000
matrices = list()
for frame in imu[:k]:
    rot = frame[:54].reshape((6, 3, 3))
    acc = frame[54:].reshape((6, 3))
    matrices.append(rot)
    
    
    
XX = [100, 0, 200, 0, 200, 100]
YY = [200, 100, 100, 300, 300, 0]
    
    
import tkinter as tk
import time

# Create a function to display each frame
def display_frame(frame_data):
    # Create a tkinter window
    window = tk.Tk()
    window.title("Frame Display")

    # Create a canvas to display your data
    canvas = tk.Canvas(window, width=400, height=400)
    canvas.pack()

    # Loop through each frame and display it
    pre_time = time.time()
    for frame in frame_data:
        pre_time += 0.016
        # Clear the canvas
        canvas.delete("all")

        # Display each matrix in the frame
        for i in range(6):
            for row in range(3):
                for col in range(3):
                    value = frame[i][row][col]
                    canvas.create_text(
                        XX[i] + col * 40 +50,
                        YY[i] + row * 20 +30,
                        text="{:.2f}".format(value),
                        font=5
                    )

        # Update the window
        while time.time() < pre_time:
            time.sleep(0.001)
        window.update()

    # window.mainloop()


# Call the function to display frames
display_frame(matrices)
