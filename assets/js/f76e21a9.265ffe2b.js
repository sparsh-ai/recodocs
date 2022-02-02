"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[860],{3905:function(e,t,n){n.d(t,{Zo:function(){return p},kt:function(){return f}});var o=n(67294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);t&&(o=o.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,o)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,o,r=function(e,t){if(null==e)return{};var n,o,r={},i=Object.keys(e);for(o=0;o<i.length;o++)n=i[o],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(o=0;o<i.length;o++)n=i[o],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var c=o.createContext({}),s=function(e){var t=o.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},p=function(e){var t=s(e.components);return o.createElement(c.Provider,{value:t},e.children)},m={inlineCode:"code",wrapper:function(e){var t=e.children;return o.createElement(o.Fragment,{},t)}},d=o.forwardRef((function(e,t){var n=e.components,r=e.mdxType,i=e.originalType,c=e.parentName,p=l(e,["components","mdxType","originalType","parentName"]),d=s(n),f=r,b=d["".concat(c,".").concat(f)]||d[f]||m[f]||i;return n?o.createElement(b,a(a({ref:t},p),{},{components:n})):o.createElement(b,a({ref:t},p))}));function f(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var i=n.length,a=new Array(i);a[0]=d;var l={};for(var c in t)hasOwnProperty.call(t,c)&&(l[c]=t[c]);l.originalType=e,l.mdxType="string"==typeof e?e:r,a[1]=l;for(var s=2;s<i;s++)a[s]=n[s];return o.createElement.apply(null,a)}return o.createElement.apply(null,n)}d.displayName="MDXCreateElement"},46744:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return l},contentTitle:function(){return c},metadata:function(){return s},assets:function(){return p},toc:function(){return m},default:function(){return f}});var o=n(87462),r=n(63366),i=(n(67294),n(3905)),a=["components"],l={title:"Object detection with YOLO3",authors:["sparsh"],tags:["app","vision","streamlit"]},c=void 0,s={permalink:"/blog/2021/01/23/object-detection-with-yolo3",editUrl:"https://github.com/recohut/docs/blog/blog/2021-01-23-object-detection-with-yolo3.mdx",source:"@site/blog/2021-01-23-object-detection-with-yolo3.mdx",title:"Object detection with YOLO3",description:"Live app",date:"2021-01-23T00:00:00.000Z",formattedDate:"January 23, 2021",tags:[{label:"app",permalink:"/blog/tags/app"},{label:"vision",permalink:"/blog/tags/vision"},{label:"streamlit",permalink:"/blog/tags/streamlit"}],readingTime:1.975,truncated:!1,authors:[{name:"Sparsh Agarwal",title:"Principal Developer",url:"https://github.com/sparsh-ai",imageURL:"https://avatars.githubusercontent.com/u/62965911?v=4",key:"sparsh"}],prevItem:{title:"What is Livestream Ecommerce",permalink:"/blog/2021/10/01/what-is-livestream-ecommerce"},nextItem:{title:"MobileNet SSD Caffe Pre-trained model",permalink:"/blog/2020/01/19/mobilenet-ssd-caffe-pre-trained-model"}},p={authorsImageUrls:[void 0]},m=[{value:"Live app",id:"live-app",children:[],level:2},{value:"Code",id:"code",children:[],level:2}],d={toc:m};function f(e){var t=e.components,l=(0,r.Z)(e,a);return(0,i.kt)("wrapper",(0,o.Z)({},d,l,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h2",{id:"live-app"},"Live app"),(0,i.kt)("p",null,"This app can detect COCO 80-classes using three different models - Caffe MobileNet SSD, Yolo3-tiny, and Yolo3. It can also detect faces using two different models - SSD Res10 and OpenCV face detector.  Yolo3-tiny can also detect fires."),(0,i.kt)("p",null,(0,i.kt)("img",{alt:"/img/content-blog-raw-blog-object-detection-with-yolo3-untitled.png",src:n(5900).Z})),(0,i.kt)("p",null,(0,i.kt)("img",{alt:"/img/content-blog-raw-blog-object-detection-with-yolo3-untitled-1.png",src:n(67649).Z})),(0,i.kt)("h2",{id:"code"},"Code"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'import streamlit as st\nimport cv2\nfrom PIL import Image\nimport numpy as np\nimport os\n\nfrom tempfile import NamedTemporaryFile\nfrom tensorflow.keras.preprocessing.image import img_to_array, load_img\n\ntemp_file = NamedTemporaryFile(delete=False)\n\nDEFAULT_CONFIDENCE_THRESHOLD = 0.5\nDEMO_IMAGE = "test_images/demo.jpg"\nMODEL = "model/MobileNetSSD_deploy.caffemodel"\nPROTOTXT = "model/MobileNetSSD_deploy.prototxt.txt"\n\nCLASSES = [\n    "background",\n    "aeroplane",\n    "bicycle",\n    "bird",\n    "boat",\n    "bottle",\n    "bus",\n    "car",\n    "cat",\n    "chair",\n    "cow",\n    "diningtable",\n    "dog",\n    "horse",\n    "motorbike",\n    "person",\n    "pottedplant",\n    "sheep",\n    "sofa",\n    "train",\n    "tvmonitor",\n]\nCOLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))\n\n@st.cache\ndef process_image(image):\n    blob = cv2.dnn.blobFromImage(\n        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5\n    )\n    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)\n    net.setInput(blob)\n    detections = net.forward()\n    return detections\n\n@st.cache\ndef annotate_image(\n    image, detections, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD\n):\n    # loop over the detections\n    (h, w) = image.shape[:2]\n    labels = []\n    for i in np.arange(0, detections.shape[2]):\n        confidence = detections[0, 0, i, 2]\n\n        if confidence > confidence_threshold:\n            # extract the index of the class label from the `detections`,\n            # then compute the (x, y)-coordinates of the bounding box for\n            # the object\n            idx = int(detections[0, 0, i, 1])\n            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])\n            (startX, startY, endX, endY) = box.astype("int")\n\n            # display the prediction\n            label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"\n            labels.append(label)\n            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)\n            y = startY - 15 if startY - 15 > 15 else startY + 15\n            cv2.putText(\n                image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2\n            )\n    return image, labels\n\ndef main():\n  selected_box = st.sidebar.selectbox(\n    \'Choose one of the following\',\n    (\'Welcome\', \'Object Detection\')\n    )\n    \n  if selected_box == \'Welcome\':\n      welcome()\n  if selected_box == \'Object Detection\':\n      object_detection() \n\ndef welcome():\n  st.title(\'Object Detection using Streamlit\')\n  st.subheader(\'A simple app for object detection\')\n  st.image(\'test_images/demo.jpg\',use_column_width=True)\n\ndef object_detection():\n  \n  st.title("Object detection with MobileNet SSD")\n\n  confidence_threshold = st.sidebar.slider(\n    "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)\n\n  st.sidebar.multiselect("Select object classes to include",\n  options=CLASSES,\n  default=CLASSES\n  )\n\n  img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])\n\n  if img_file_buffer is not None:\n      temp_file.write(img_file_buffer.getvalue())\n      image = load_img(temp_file.name)\n      image = img_to_array(image)\n      image = image/255.0\n\n  else:\n      demo_image = DEMO_IMAGE\n      image = np.array(Image.open(demo_image))\n\n  detections = process_image(image)\n  image, labels = annotate_image(image, detections, confidence_threshold)\n\n  st.image(\n      image, caption=f"Processed image", use_column_width=True,\n  )\n\n  st.write(labels)\n\nmain()\n')),(0,i.kt)("p",null,(0,i.kt)("em",{parentName:"p"},"You can play with the live app")," ",(0,i.kt)("a",{parentName:"p",href:"https://share.streamlit.io/sparsh-ai/streamlit-489fbbb7/app.py"},"*here"),". Source code is available ",(0,i.kt)("a",{parentName:"p",href:"https://github.com/sparsh-ai/streamlit-5a407279/tree/master"},"here")," on Github.*"))}f.isMDXComponent=!0},67649:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-object-detection-with-yolo3-untitled-1-c0c4b05a1cf3256f1a9d0ebdb5f4bfb5.png"},5900:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-object-detection-with-yolo3-untitled-7c24204b83a8bd10c57c53b4b2423899.png"}}]);