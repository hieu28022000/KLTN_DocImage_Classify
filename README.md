# KLTN_DocImage_Classify

## Mô tả chung   
Đây là dự án **Phân loại hình ảnh văn bản** cho các loại tài liệu **Báo cáo Tài chính Ngân hàng**. Dự án sử dụng các mô hình khác nhau nhằm đánh giá kết quả đầu ra của các mô hình trên bộ dữ liệu Báo cáo Tài chính Ngân hàng. Các mô hình được sử dụng là:   
    
- Visual and Textual Feature - VGG16
- Visual and Textual Feature - ResNet50
- Visual and Textual Feature - Xception
- Vision Transformer
- Bert
- LayoutLM

## Cài đặt   
Vì dự án có sử dụng mô hình OCR đào tạo trước để trích xuất thông tin văn bản từ hình ảnh nên sẽ cần cài đặt 2 môi trường khác nhau để thực hiện dồng thời.


### Clone source:  
```
git clone https://github.com/hieu28022000/KLTN_DocImage_Classify.git   
cd KLTN_DocImage_Classify/   
```   

### Cài đặt môi trường cho mô hình ocr đào tạo trước:   
- Download [soc model]() và đặt vào ```./models/bert/```
```
docker build -t ocr_image -f Dockerfile_ocr .   
docker run -it -d --name ocr -p 81:80 ocr_image
```   

### Cài đặt môi trường cho dự án
- Cập nhật lại ```url_ocr``` trong ```configs/api_config.py``` (Host của url chính là IPAddress của container ocr đượ tạo ở trên - ```docker inspect ocr```)  
- Download [VGG16 model]() và đặt vào ```./models/vatf/backup/```
- Download [ResNet50 model]() và đặt vào ```./models/vatf/backup/```
- Download [Xception model]() và đặt vào ```./models/vatf/backup/```
- Download [Vision Transformer weight]() và đặt vào ```./models/vision_transformer/backup/```
- Download [Bert weight]() và đặt vào ```./models/bert/backup/```
- Download [LayoutLM model]() và đặt vào ```./models/layoutlm/backup/```
- Download [VGG16 model]() và đặt vào ```./models/vatf/backup/```
```
docker build -t cls_image -f Dockerfile .   
docker run -it -d --name classify -p 80:80 cls_image
```   

### Thực nghiệm
- Truy cập http://0.0.0.0:80/home mở web ứng dụng của dự án.
- Chọn tên mô hình cần sử dụng.
- Upload hình ảnh thực nghiệm.
- Bấm ```classify``` để bắt đầu xử lý.
- Kết quả sẽ hiển thị ben dưới bao gồm thông tin loại văn bản và điểm tin cậy của mô hình.

## Liên hệ
Nguyễn Quang Hiếu   
Gmail: 18520748@gm.uit.edu.vn
