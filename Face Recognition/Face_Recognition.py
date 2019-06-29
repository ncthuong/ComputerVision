
# coding: utf-8

# # **Nhận dạng khuôn mặt với OpenCV và Python**
# 
# ***Sơ lược***
# 
# Trong cuộc sống thường ngày, khi chúng ta nhìn thấy một vật thể nào đó, chúng ta hoàn toàn có khả năng nhận ra đó là cái gì nếu chúng ta đã biết nó trước đó. Ví dụ, khi nhìn thấy chiếc xe hơi, chúng ta có thể chắc chắn kết luận đó là một chiếc xe hơi chứ không phải là xe đạp hay xe máy, hoặc khi thấy người thân trong một bức ảnh, chúng ta cũng có thể dễ dàng nhận ra.
# 
# Tại sao não chúng ta làm được chuyện đó? Câu trả lời là bởi vì não chúng ta đã được "học" và ghi nhận về thứ đó trước đây trong quá khứ, đến khi gặp lại việc nhận ra không có gì khó khăn. Ví dụ, nếu chúng ta gặp một người tên Nam trong lần đầu gặp mặt, não chúng ta sẽ tập trung ghi nhận các dữ liệu về khuôn mặt trước tiên như: mắt, mũi, miệng, màu da và tổng quan khuôn mặt. Sau đó, não chúng ta sẽ học và ghi nhận rằng với dữ liệu này thì xác định đây chính là Nam. Nghe có vẻ quá đơn giản, nhưng thực tế điều đó chỉ đơn giản với não con người. Đối với máy tính, phải làm sao để nó có thể nhận dạng ra được một người nào đó trong một ảnh hay một video là điều rất khó khăn.
# 
# Ngày nay, dưới sự ra đời và hỗ trợ của các thư viện, chúng ta có thể dàng "làm" cho máy tính thực hiện được việc nhận dạng khuôn mặt một các dễ dàng. Một trong số đó là thư viện OpenCV. Đây là một thư viện mã nguồn mở hàng đầu cho ngành thị giác máy tính. OpenCV được phát hành theo giấy phép BSD, do đó nó hoàn toàn miễn phí cho cả học thuật và thương mại. Nó có các interface C++, C, Python, Java và hỗ trợ trên các nền tảng Windows, Linux, MacOS, iOS và Android. OpenCV được thiết kế để tính toán hiệu quả đối với các dụng theo thời gian thực, tận dụng lợi thế xử lý đa lõi thông qua việc được viết bằng C/C++. Cộng đồng sử dụng OpenCV hơn 47 nghìn người và số lượng download vượt quá 6 triệu lần.
# 
# OpenCV Face Recognizers
# 
# OpenCV có 3 Face Recognizers với code rất rõ ràng bao gồm:
# 1. EigenFaces Face Recognizer Recognizer -cv2.face.createEigenFaceRecognizer()
# 2. FisherFaces Face Recognizer Recognizer - cv2.face.createFisherFaceRecognizer()
# 3. Local Binary Patterns Histograms (LBPH) Face Recognizer - cv2.face.createLBPHFaceRecognizer()
# 

#  **Xây dựng Nhận dạng khuôn mặt với OpenCV**
# 
# Qúa trình xây dựng bộ nhận dạng khuôn mặt bao gồm 3 bước:
# 
# 1. Chuẩn bị dữ liệu: trong bước này chúng ta sẽ đọc dữ liệu training cho mỗi đối tượng với nhãn của đối tượng đó, phát hiện khuôn mặt từ mỗi ảnh và gán mỗi khuôn mặt phát hiện được với một nhãn là số nguyên.
# 2. Đào tạo bộ nhận dạng khuôn mặt: trong bước này chúng ra sẽ đào tạo bộ nhận dạng LBPH của OpenCV bằng dữ liệu đã được chuẩn bị ở bước 1.
# 3. Kiểm tra: trong bước này chúng ta sẽ kiểm tra xem bộ nhận dạng đã đoán đối tượng trong ảnh test đã chính xác chưa.
# 

# In[51]:


#import OpenCV
import cv2
#import module os cho việc đọc thư mục và đường dẫn dữ liệu training
import os
#import numoy để chuyển các list trong python thành các mảng numpy
import numpy as np


# Traning Data
# 
# Dữ liệu training càng nhiều càng tốt. Thông thường nhiều ảnh được sử dụng để training bộ nhận dạng khuôn mặt với các góc ảnh và trạng thái của khác nhau của khuôn mặt như đeo kính, không đeo kính, đang cười, buồn, hạnh phúc, đang khóc, có râu, không có râu... Trong bài này, để đơn giản chúng ta chỉ sử dụng 12 ảnh cho mỗi người.
# Dữ liệu training cho 2 người với mỗi người có 12 ảnh và nằm trong tệp training-data. Cây cấu trúc thư mục dữ liệu training như sau:
# 
# training-data
# 
# |-------------- s1
# 
# |               |-- 1.jpg
# 
# |               |-- ...
# 
# |               |-- 12.jpg
# 
# |-------------- s2
# 
# |               |-- 1.jpg
# 
# |               |-- ...
# 
# |               |-- 12.jpg
# 
# 
# Thư mục test-data bao gồm các ảnh sẽ được sử dụng để test bộ nhận dạng khuôn mặt sau khi đã training thành công.
# 
# vì bộ nhận dạng khuôn mặt OpenCV chấp nhận các nhãn là số nguyên, nên chúng ta cần xác định ánh xạ giữa nhãn số nguyên và tên thực của người, vì vậy bên dưới chúng ta sẽ xác định ánh xạ của nhãn số nguyên và tên tương ứng của chúng.
#  
#  Note: Vì chúng ta không có gán nhãn 0 đến bất kỳ người nào nên ánh xạ cho nhãn 0 là rỗng.

# In[52]:


#Không có nhãn 0 trong tập dữ liệu training vì vậy tên cho chỉ số/thư mục 0 là rỗng
subjects = ["","Tran Thanh", "Hoai Linh"]


# **Chuẩn bị dữ liệu training**
# 
# Tại sao chúng ta cần chuẩn bị dữ liệu? Bởi vì bộ nhận dạng khuôn mặt của OpenCV chấp nhận dữ liệu trong định dạng vector. Một vector là khuôn mặt của tất các các người và vector thứ hai là nhãn số nguyên cho mỗi khuôn mặt, để khi xử lý một khuôn mặt, bộ nhận diện biết được khuôn mặt đó là của người nào.
# 
# Ví dụ: giả sử chúng ta có 2 người và 2 ảnh cho mỗi người
# 
# PERSON-1    PERSON-2   
# 
# img1        img1         
# img2        img2
# 
# Thì khi chuẩn bị dữ liệu, chúng ra cần tạo ra vector mặt và nhãn như sau:
# 
# FACES                        LABELS
# 
# person1_img1_face              1
# person1_img2_face              1
# person2_img1_face              2
# person2_img2_face              2
# 
# Bước chuẩn bị dữ liệu có thể chia thành các bước con sau:
# 
# 1. Đọc tất cả tên thư mục của đối tượng/người cung cấp trong thư mục dữ liệu training. ví dụ: trong bài này chúng ta có tên thư mục là s1, s2.
# 
# 2. Vỡi mỗi đối tượng, trích xuất số nhãn. Tên thư mục có định dạng sLabel trong đó Label là một số nguyên thể hiện nhãn mà chúng ta đã gáng cho đối tượng. Ví dụ: tên thư mục là s1 có nghĩa là đối tượng có nhãn là 1, s2 có nghĩa là đối tượng có nhãn là 2... Nhãn được trích xuất trong bước này được gán cho mỗi khuôn mặt được phát hiện trong bước kế tiếp.
# 
# 3. Đọc tất cả các ảnh của đối tượng, phát hiện khuôn mặt trong mỗi ảnh.
# 
# 4. Thêm mỗi mặt được phát hiện vào vector mặt với nhãn đối tượng tương ứng thì được thêm vào vector nhãn

# In[53]:


#function để phát hiện khuôn mặt sử dụng OpenCV
def detect_face(img):
    #chuyển đổi ảnh test thành ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Load bộ phát hiện mặt của OpenCV, sử dụng LBP vì nó nhanh và độc chính xác cao hơn Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files\lbpcascade_frontalface.xml')
    
    #Phát hiện nhiều ảnh (một vài ảnh có thể gần camera hơn những ảnh khác)
    #Kết quả là một danh sách các khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5);
    
    #nếu không có khuôn mặt nào được phát hiện thì quay lại ảnh gốc
    if (len(faces)==0):
        return None, None
    
    #Chúng ta giả định rằng mỗi ảnh chỉ có một khuôn mặt
    #trích xuất vùng khuôn mặt
    (x, y, w, h) = faces[0]
    
    #chỉ trả về phần khuôn mặt của ảnh
    return gray[y:y+w, x:x+h], faces[0] 
    


# In[54]:


#Function này sẽ đọc tất cả các ảnh training người, phát hiện khuôn mặt trong mỗi ảnh
#và sẽ trả về 2 danh sách có cùng kích thước, một danh sách các khuôn mặt và một danh sách các nhãn của mỗi khuôn mặt
def prepare_training_data(data_folder_path):
    #-----Bước 1------
    #lấy địa chỉ thư mục (một thư mục cho mỗi đối tượng) trong data folder
    dirs = os.listdir(data_folder_path)
    
    #danh sách để chứa tất cả khuôn mặt các đối tượng
    faces = []
    #danh sách để chứa các nhãn cho tất cả đối tượng
    labels = []
    
    #Duyệt qua mỗi thư mục và đọc các ảnh trong đó
    for dir_name in dirs:
        #Thư mục các đối tượng bắt đầu bằng ký tự "s" vì vậy bỏ qua mọi thư mục không liên quan nếu có
        if not dir_name.startswith("s"):
            continue;
         
        #----Bước 2-----
        #Trích xuất số nhãn của đối tượng từ dir_name
        #định dạng của dir name = slabel, vì vậy xóa ký tự "s" từ dir_name chúng ta sẽ được nhãn
        label = int(dir_name.replace("s",""))
        
        #xây dựng đường dẫn của thư mục hiện tại chứa ảnh đối tượng hiện tại
        #ví dụ subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #lấy tên các ảnh nằm trong thư mục đối tượng
        subject_images_names = os.listdir(subject_dir_path)
        
        #----Bước 3-----
        #Duyệt qua mỗi ảnh, đọc ảnh, phát hiện khuôn mặt và thêm vào danh sách khuôn mặt
        for image_name in subject_images_names:
            #bỏ qua các file hệ thống như .DS_Store
            if image_name.startswith("."):
                continue;
            
            #xây dựng đường dẫn, ví dụ path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name
            
            #đọc ảnh
            image = cv2.imread(image_path)
            
            #hiển thị một cửa sổ ảnh
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
            
            #Nhận diện khuôn mặt
            face, rect = detect_face(image)
            
            #----Bước 4-----
            #Trong phạm vi bài này chúng ta bỏ qua các khuôn mặt không được phát hiện
            if face is not None:
                #Thêm khuôn mặt vào danh sách
                faces.append(face)
                #thêm nhãn cho khuôn mặt đó
                labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels
 
                
    


# In[55]:


#Đầu tiên cần chuẩn bị dữ liệu training
#dữ liệu có trong hai danh sách cùng kích thước
#một danh sách bao gồm tất cả khuôn mặt
#Một danh sách bao gồm các nhãn cho mỗi khuôn mặt
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#in ra tổng các khuôn mặt và lable
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


# Training bộ nhận dạng khuôn mặt
# 
# Chúng ta đã biết OpenCV có ba bộ nhận dạng khuôn  mặt:
# 
# 1. EigenFace Recognizer:
# cv2.face.createEigenFaceRecognizer()
# 2. FisherFace Recognizer:
# cv2.face.createFisherFaceRecognizer()
# 3. Local Binary Patterns Histogram (LBPH):
# cv2.face.LBPHFisherFaceRecognizer()
# 
# Chúng ta sẽ sử dụng LBPH nhưng bạn có thể sử dụng bộ nhận dạng nào của OpenCV đều được. Đơn giản chỉ cần thay đổi dòng khởi tạo bộ nhận dạng
# 

# In[56]:


#tạo bộ nhận LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#đào tạo bộ nhận khuôn mặt 
face_recognizer.train(faces, np.array(labels))


# **Dự đoán**
# 
# Để xem thực sự liệu thuật toán có nhận ra khuôn mặt của các đối tượng được training hay không. Chúng ta sẽ lấy hai ảnh test, phát hiện khuôn mặt từ chúng và sau đó chuyển chúng đến bộ nhận dạng để xem nó có nhận ra không.
# 

# In[57]:


#function để vẽ hình chữ nhật trên ảnh theo tọa độ (x, y) và chiều rộng, chiều cao
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 0)
    
#function để vẽ chữ trên ảnh bắt đầu từ tọa độ (x, y)
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

#function để nhận dạng một người trong ảnh và vẽ hình chữ nhật quanh mặt người đó với tên của họ
def predict(test_img):
    #tạo một bản copy ảnh vì chúng ta không muốn thay đổi ảnh gốc
    img = test_img.copy()
    
    #phát hiện khuon mặt từ ảnh
    face, rect = detect_face(img)
    
    #dự đoán ảnh sử dụng bộ nhận dạng khuôn mặt của chúng ta
    label = face_recognizer.predict(face)
    label = label[0]
    
    #lấy tên của nhãn tương ứng được trả về bởi bộ nhạn dạng khuôn mặt
    label_text = subjects[label]
    
    #vẽ một hình chữ nhật quanh khuôn mặt
    draw_rectangle(img, rect)
    
    #vẽ tên của người được dự đoán
    draw_text(img, label_text, rect[0], rect[1]-5)
    return img


# Bây giờ chúng ta đã xác định được hàm dự đoán, bước kế tiếp là gọi hàm này trên các ảnh test và hiển thị ảnh test để xem bộ nhận dạng của chúng ta đã nhận dạng chính xác chưa.

# In[62]:


print("Predicting images...")

#load ảnh test
test_img1 = cv2.imread('test-data/test1.jpg')
test_img2 = cv2.imread('test-data/test2.jpg')

#biểu diễn dự đoán
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("Prediction complete")

#hiển thị hai ảnh
cv2.imshow(subjects[1], predicted_img1)
cv2.imshow(subjects[2], predicted_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# **Kết quả đạt được**
# 
# Bộ nhận dạng khuôn mặt được xây dựng đã nhận dạng thành công với 2 ảnh test thử của 2 đối tượng.
# Tuy nhiên, dữ liệu còn ít nên có trường hợp ảnh test với khuôn mặt đối tượng có góc nghiêng quá lớn thì bộ nhận dạng sẽ cho ra kết quả sai.
# 

# Tài liệu tham khảo
# 
# 1. https://github.com/informramiz/opencv-face-recognition-python/blob/master/OpenCV-Face-Recognition-Python.ipynb
# 2. https://towardsdatascience.com/face-recognition-using-artificial-intelligence-fffa3b20ad5f
