import cv2
import numpy as np 
import argparse
import time

parser = argparse.ArgumentParser() # membuat argumen untuk menjalankan program yolo.py dengan argumen --webcam, --play_video, --image, --video_path, --image_path, --verbose 
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="Tue/False", default=False)
parser.add_argument('--image', help="True/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="videos/car_on_road.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args() # mengambil argumen yang diinputkan pada program yolo.py 

#Load yolo
def load_yolo(): # fungsi untuk load yolo dengan menggunakan file weights dan file konfigurasi yang sudah disediakan 
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") #membaca file weights dan file konfigurasi yang sudah disediakan
	#yolov3.weights adalah file model yang sudah dilatih sebelumnya menggunakan dataset COCO
	#yolov3.cfg adalah file konfigurasi yang digunakan untuk meload model yang sudah dilatih sebelumnya dalam file yolov3.weights
	classes = [] # membuat list kosong untuk menyimpan nama kelas yang ada pada dataset COCO
	with open("coco.names", "r") as f: # membuka file coco.names yang berisi nama kelas yang ada pada dataset COCO 
		classes = [line.strip() for line in f.readlines()]  # menyimpan nama kelas yang ada pada dataset COCO ke dalam list classes
	
	output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()] # menyimpan nama layer yang ada pada model yang sudah dilatih sebelumnya ke dalam list output_layers
	#net.getUnconnectedOutLayersNames() adalah fungsi untuk mendapatkan nama layer yang ada pada model yang sudah dilatih sebelumnya

	# membuat array 2 dimensi dengan ukuran (jumlah kelas, 3) yang berisi bilangan random dari 0 sampai 255 untuk digunakan sebagai warna bounding box yang akan digunakan untuk menandai objek yang terdeteksi
	colors = np.random.uniform(0, 255, size=(len(classes), 3))

	return net, classes, colors, output_layers # mengembalikan nilai net, classes, colors, output_layers yang akan digunakan pada fungsi selanjutnya 

def load_image(img_path): # fungsi untuk load image yang akan digunakan untuk mendeteksi objek 

	# image loading
	img = cv2.imread(img_path) # membaca image yang akan digunakan untuk mendeteksi objek
	img = cv2.resize(img, None, fx=0.4, fy=0.4) # mengubah ukuran image menjadi 40% dari ukuran aslinya
	height, width, channels = img.shape # mengambil ukuran image yang sudah diubah ukurannya
	return img, height, width, channels # mengembalikan nilai img, height, width, channels yang akan digunakan pada fungsi selanjutnya 

def start_webcam(): # fungsi untuk memulai webcam 
	cap = cv2.VideoCapture(0) # membuka webcam dengan index 0 (default webcam), index 1 (eksternal webcam)
	return cap # mengembalikan nilai cap yang akan digunakan pada fungsi selanjutnya

# Blob (binary large object) adalah sebuah image yang diubah menjadi array multidimensi yang berisi nilai pixel dari image tersebut 
def display_blob(blob): # fungsi untuk menampilkan blob 
	'''
		Three images each for RED, GREEN, BLUE channel
	'''
	for b in blob: # melakukan iterasi pada blob 
		for n, imgb in enumerate(b): # melakukan iterasi pada blob 
			cv2.imshow(str(n), imgb) # menampilkan blob 

def detect_objects(img, net, outputLayers):	 # fungsi untuk mendeteksi objek pada image 		
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False) # mengubah image menjadi blob 
	net.setInput(blob) # menginputkan blob ke dalam model yang sudah dilatih sebelumnya 
	outputs = net.forward(outputLayers) # melakukan forward propagation pada model yang sudah dilatih sebelumnya dengan input blob 
	return blob, outputs # mengembalikan nilai blob, outputs yang akan digunakan pada fungsi selanjutnya 

def get_box_dimensions(outputs, height, width): # fungsi untuk mendapatkan koordinat bounding box dari objek yang terdeteksi 
	boxes = [] # membuat list kosong untuk menyimpan koordinat bounding box dari objek yang terdeteksi
	confs = [] # membuat list kosong untuk menyimpan confidence dari objek yang terdeteksi
	class_ids = [] # membuat list kosong untuk menyimpan class id dari objek yang terdeteksi
	for output in outputs: 
		for detect in output:
			scores = detect[5:] # mengambil nilai confidence dari objek yang terdeteksi, confidence = nilai probabilitas dari objek yang terdeteksi
			class_id = np.argmax(scores) # mengambil class id dari objek yang terdeteksi
			conf = scores[class_id] # mengambil nilai confidence dari objek yang terdeteksi 
			if conf > 0.2: # jika nilai confidence lebih dari 0.2 maka objek tersebut akan dideteksi 
				center_x = int(detect[0] * width) # mengambil nilai center x dari objek yang terdeteksi 
				center_y = int(detect[1] * height) # mengambil nilai center y dari objek yang terdeteksi 
				w = int(detect[2] * width) # mengambil nilai width dari objek yang terdeteksi 
				h = int(detect[3] * height) # mengambil nilai height dari objek yang terdeteksi
				x = int(center_x - w/2) # mengambil nilai x dari objek yang terdeteksi 
				y = int(center_y - h / 2) # mengambil nilai y dari objek yang terdeteksi
				boxes.append([x, y, w, h]) # menyimpan nilai x, y, w, h dari objek yang terdeteksi ke dalam list boxes
				confs.append(float(conf)) # menyimpan nilai confidence dari objek yang terdeteksi ke dalam list confs
				class_ids.append(class_id) # menyimpan nilai class id dari objek yang terdeteksi ke dalam list class_ids
	return boxes, confs, class_ids # mengembalikan nilai boxes, confs, class_ids yang akan digunakan pada fungsi selanjutnya
			
def draw_labels(boxes, confs, colors, class_ids, classes, img): # fungsi untuk menampilkan label dari objek yang terdeteksi
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4) # fungsi untuk menghapus bounding box yang overlap dan mengambil bounding box yang memiliki nilai confidence tertinggi
	font = cv2.FONT_HERSHEY_PLAIN # membuat font untuk menampilkan label dari objek yang terdeteksi
	for i in range(len(boxes)): # melakukan iterasi pada list boxes
		if i in indexes: # jika nilai i ada di dalam list indexes maka bounding box akan ditampilkan
			x, y, w, h = boxes[i] # mengambil nilai x, y, w, h dari list boxes 
			label = str(classes[class_ids[i]]) # mengambil label dari objek yang terdeteksi 
			color = colors[i % len(colors)] # mengambil warna dari objek yang terdeteksi
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2) # menampilkan bounding box pada objek yang terdeteksi
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1) # menampilkan label pada objek yang terdeteksi
	cv2.imshow("Image", img) # menampilkan image yang sudah dideteksi objeknya 

def image_detect(img_path): # fungsi untuk mendeteksi objek pada image 
	model, classes, colors, output_layers = load_yolo() # memanggil fungsi load_yolo() dan menyimpannya ke dalam variabel model, classes, colors, output_layers
	image, height, width, channels = load_image(img_path) # memanggil fungsi load_image() dan menyimpannya ke dalam variabel image, height, width, channels
	blob, outputs = detect_objects(image, model, output_layers) # memanggil fungsi detect_objects() dan menyimpannya ke dalam variabel blob, outputs
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width) # memanggil fungsi get_box_dimensions() dan menyimpannya ke dalam variabel boxes, confs, class_ids
	draw_labels(boxes, confs, colors, class_ids, classes, image) # memanggil fungsi draw_labels()
	while True: # melakukan iterasi pada image yang sudah dideteksi objeknya
		key = cv2.waitKey(1)
		if key == 27:
			break

def webcam_detect(): # fungsi untuk mendeteksi objek pada webcam 
	model, classes, colors, output_layers = load_yolo() # memanggil fungsi load_yolo() dan menyimpannya ke dalam variabel model, classes, colors, output_layers
	cap = start_webcam() # memanggil fungsi start_webcam() dan menyimpannya ke dalam variabel cap 
	while True: # melakukan iterasi pada webcam yang sudah dideteksi objeknya
		_, frame = cap.read() # membaca frame dari webcam 
		height, width, channels = frame.shape # mengambil nilai height, width, channels dari frame webcam 
		blob, outputs = detect_objects(frame, model, output_layers) # memanggil fungsi detect_objects() dan menyimpannya ke dalam variabel blob, outputs 
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width) # memanggil fungsi get_box_dimensions() dan menyimpannya ke dalam variabel boxes, confs, class_ids
		draw_labels(boxes, confs, colors, class_ids, classes, frame) # memanggil fungsi draw_labels() 
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()


def start_video(video_path): # fungsi untuk mendeteksi objek pada video
	model, classes, colors, output_layers = load_yolo() # memanggil fungsi load_yolo() dan menyimpannya ke dalam variabel model, classes, colors, output_layers
	cap = cv2.VideoCapture(video_path) # membaca video 
	while True:
		_, frame = cap.read() # membaca frame dari video 
		height, width, channels = frame.shape # mengambil nilai height, width, channels dari frame video
		blob, outputs = detect_objects(frame, model, output_layers) # memanggil fungsi detect_objects() dan menyimpannya ke dalam variabel blob, outputs
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width) # memanggil fungsi get_box_dimensions() dan menyimpannya ke dalam variabel boxes, confs, class_ids
		draw_labels(boxes, confs, colors, class_ids, classes, frame) # memanggil fungsi draw_labels()
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()

if __name__ == '__main__':
	webcam = args.webcam # menyimpan nilai webcam dari argumen yang diberikan
	video_play = args.play_video # menyimpan nilai play_video dari argumen yang diberikan
	image = args.image # menyimpan nilai image dari argumen yang diberikan
	if webcam: # jika webcam bernilai True maka akan memanggil fungsi webcam_detect()
		if args.verbose:
			print('---- Memulai menyalakan Webcam device.. ----') 
		webcam_detect() # memanggil fungsi webcam_detect()
	if video_play: # jika video_play bernilai True maka akan memanggil fungsi start_video()
		video_path = args.video_path # menyimpan nilai video_path dari argumen yang diberikan
		if args.verbose:
			print('Membuka video '+video_path+" .... ")
		start_video(video_path) # memanggil fungsi start_video()
	if image: # jika image bernilai True maka akan memanggil fungsi image_detect()
		image_path = args.image_path # menyimpan nilai image_path dari argumen yang diberikan
		if args.verbose:
			print("Membuka foto "+image_path+" .... ")
		image_detect(image_path) # memanggil fungsi image_detect() 
	

	cv2.destroyAllWindows() # menutup semua window yang terbuka
