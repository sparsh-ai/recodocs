"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[6018],{3905:function(e,t,n){n.d(t,{Zo:function(){return p},kt:function(){return b}});var r=n(67294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var s=r.createContext({}),l=function(e){var t=r.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},p=function(e){var t=l(e.components);return r.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,a=e.originalType,s=e.parentName,p=c(e,["components","mdxType","originalType","parentName"]),d=l(n),b=o,h=d["".concat(s,".").concat(b)]||d[b]||u[b]||a;return n?r.createElement(h,i(i({ref:t},p),{},{components:n})):r.createElement(h,i({ref:t},p))}));function b(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=n.length,i=new Array(a);i[0]=d;var c={};for(var s in t)hasOwnProperty.call(t,s)&&(c[s]=t[s]);c.originalType=e,c.mdxType="string"==typeof e?e:o,i[1]=c;for(var l=2;l<a;l++)i[l]=n[l];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},4111:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return c},contentTitle:function(){return s},metadata:function(){return l},assets:function(){return p},toc:function(){return u},default:function(){return b}});var r=n(87462),o=n(63366),a=(n(67294),n(3905)),i=["components"],c={title:"OCR experiments",authors:"sparsh",tags:["ocr","vision"]},s=void 0,l={permalink:"/blog/2021/10/01/ocr-experiments",editUrl:"https://github.com/recohut/docs/blog/blog/2021-10-01-ocr-experiments.mdx",source:"@site/blog/2021-10-01-ocr-experiments.mdx",title:"OCR experiments",description:"/img/content-blog-raw-blog-ocr-experiments-untitled.png",date:"2021-10-01T00:00:00.000Z",formattedDate:"October 1, 2021",tags:[{label:"ocr",permalink:"/blog/tags/ocr"},{label:"vision",permalink:"/blog/tags/vision"}],readingTime:1.155,truncated:!1,authors:[{name:"Sparsh Agarwal",title:"Principal Developer",url:"https://github.com/sparsh-ai",imageURL:"https://avatars.githubusercontent.com/u/62965911?v=4",key:"sparsh"}],prevItem:{title:"Object detection with YOLO3",permalink:"/blog/2021/10/01/object-detection-with-yolo3"},nextItem:{title:"PDF to Wordcloud via Mail",permalink:"/blog/2021/10/01/pdf-to-wordcloud-via-mail"}},p={authorsImageUrls:[void 0]},u=[{value:"1. Tesseract",id:"1-tesseract",children:[],level:2},{value:"2. EasyOCR",id:"2-easyocr",children:[],level:2},{value:"3. KerasOCR",id:"3-kerasocr",children:[],level:2},{value:"4. ArabicOCR",id:"4-arabicocr",children:[],level:2}],d={toc:u};function b(e){var t=e.components,c=(0,o.Z)(e,i);return(0,a.kt)("wrapper",(0,r.Z)({},d,c,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("p",null,(0,a.kt)("img",{alt:"/img/content-blog-raw-blog-ocr-experiments-untitled.png",src:n(22822).Z})),(0,a.kt)("h2",{id:"1-tesseract"},"1. Tesseract"),(0,a.kt)("p",null,"Tesseract is an open-source text recognition engine that is available under the Apache 2.0 license and its development has been sponsored by Google since 2006."),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://nbviewer.jupyter.org/gist/sparsh-ai/2d1f533048a3655de625298c3dd32d47"},"Notebook on nbviewer")),(0,a.kt)("h2",{id:"2-easyocr"},"2. EasyOCR"),(0,a.kt)("p",null,"Ready-to-use OCR with 70+ languages supported including Chinese, Japanese, Korean and Thai. EasyOCR is built with Python and Pytorch deep learning library, having a GPU could speed up the whole process of detection. The detection part is using the CRAFT algorithm and the Recognition model is CRNN. It is composed of 3 main components, feature extraction (we are currently using Resnet), sequence labelling (LSTM) and decoding (CTC). EasyOCR doesn\u2019t have much software dependencies, it can directly be used with its API."),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://nbviewer.jupyter.org/gist/sparsh-ai/12359606ee4127513c66fc3b4ff18e5b"},"Notebook on nbviewer")),(0,a.kt)("h2",{id:"3-kerasocr"},"3. KerasOCR"),(0,a.kt)("p",null,"This is a slightly polished and packaged version of the Keras CRNN implementation and the published CRAFT text detection model. It provides a high-level API for training a text detection and OCR pipeline and out-of-the-box OCR models, and an end-to-end training pipeline to build new OCR models."),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://nbviewer.jupyter.org/gist/sparsh-ai/2fcb764619baf5f56cf7122b1b2c527c"},"Notebook on nbviewer")),(0,a.kt)("h2",{id:"4-arabicocr"},"4. ArabicOCR"),(0,a.kt)("p",null,"It is an OCR system for the Arabic language that converts images of typed text to machine-encoded text. It currently supports only letters (29 letters).  ArabicOCR aims to solve a simpler problem of OCR with images that contain only Arabic characters (check the dataset link below to see a sample of the images)."),(0,a.kt)("p",null,(0,a.kt)("a",{parentName:"p",href:"https://nbviewer.jupyter.org/gist/sparsh-ai/26df76b78f8cd2018a068b284b7cfe56"},"Notebook on nbviewer")))}b.isMDXComponent=!0},22822:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-ocr-experiments-untitled-7efe4530732678d915b7176ba9205352.png"}}]);