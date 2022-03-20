"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[8006],{3905:function(e,t,n){n.d(t,{Zo:function(){return h},kt:function(){return p}});var a=n(67294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function r(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,a,i=function(e,t){if(null==e)return{};var n,a,i={},o=Object.keys(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var l=a.createContext({}),m=function(e){var t=a.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):r(r({},t),e)),n},h=function(e){var t=m(e.components);return a.createElement(l.Provider,{value:t},e.children)},c={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},d=a.forwardRef((function(e,t){var n=e.components,i=e.mdxType,o=e.originalType,l=e.parentName,h=s(e,["components","mdxType","originalType","parentName"]),d=m(n),p=i,u=d["".concat(l,".").concat(p)]||d[p]||c[p]||o;return n?a.createElement(u,r(r({ref:t},h),{},{components:n})):a.createElement(u,r({ref:t},h))}));function p(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var o=n.length,r=new Array(o);r[0]=d;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s.mdxType="string"==typeof e?e:i,r[1]=s;for(var m=2;m<o;m++)r[m]=n[m];return a.createElement.apply(null,r)}return a.createElement.apply(null,n)}d.displayName="MDXCreateElement"},8638:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return s},contentTitle:function(){return l},metadata:function(){return m},assets:function(){return h},toc:function(){return c},default:function(){return p}});var a=n(87462),i=n(63366),o=(n(67294),n(3905)),r=["components"],s={title:"Real-time news personalization with Flink",authors:"sparsh",tags:["personalization","realtime"]},l=void 0,m={permalink:"/blog/2021/10/01/real-time-news-personalization-with-flink",editUrl:"https://github.com/recohut/docs/blog/blog/2021-10-01-real-time-news-personalization-with-flink.mdx",source:"@site/blog/2021-10-01-real-time-news-personalization-with-flink.mdx",title:"Real-time news personalization with Flink",description:"Overview",date:"2021-10-01T00:00:00.000Z",formattedDate:"October 1, 2021",tags:[{label:"personalization",permalink:"/blog/tags/personalization"},{label:"realtime",permalink:"/blog/tags/realtime"}],readingTime:9.82,truncated:!1,authors:[{name:"Sparsh Agarwal",title:"Principal Developer",url:"https://github.com/sparsh-ai",imageURL:"https://avatars.githubusercontent.com/u/62965911?v=4",key:"sparsh"}],prevItem:{title:"Predicting Electronics Resale Price",permalink:"/blog/2021/10/01/predicting-electronics-resale-price"},nextItem:{title:"Semantic Similarity",permalink:"/blog/2021/10/01/semantic-similarity"}},h={authorsImageUrls:[void 0]},c=[{value:"Overview",id:"overview",children:[{value:"Why is the real-time nature of the recommendation system important?",id:"why-is-the-real-time-nature-of-the-recommendation-system-important",children:[],level:3},{value:"The real-time nature of the &quot;feature&quot; of the recommendation system",id:"the-real-time-nature-of-the-feature-of-the-recommendation-system",children:[],level:3},{value:"The real-time nature of the &quot;model&quot; of the recommender system",id:"the-real-time-nature-of-the-model-of-the-recommender-system",children:[],level:3}],level:2},{value:"Data pipeline of a typical news recommendation system",id:"data-pipeline-of-a-typical-news-recommendation-system",children:[{value:"Problems",id:"problems",children:[],level:3}],level:2},{value:"Apache Flink to the rescue",id:"apache-flink-to-the-rescue",children:[{value:"How Apache Flink solves the latency problem?",id:"how-apache-flink-solves-the-latency-problem",children:[],level:3},{value:"How Apache Flink solved the boundary and synchronization problem?",id:"how-apache-flink-solved-the-boundary-and-synchronization-problem",children:[],level:3}],level:2},{value:"References",id:"references",children:[],level:2}],d={toc:c};function p(e){var t=e.components,s=(0,i.Z)(e,r);return(0,o.kt)("wrapper",(0,a.Z)({},d,s,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h2",{id:"overview"},"Overview"),(0,o.kt)("p",null,"News recommendation system has a high degree of real-time because there will be a large number of news and hot spots at any time. Incremental updating, online learning, local updating and even reinforcement learning can make the recommender system quickly respond to the user\u2018s new behavior, and the premise of these updating strategies is that the sample itself has enough real-time information. In news recommendation system, the typical training sample is the user\u2019s click behavior data."),(0,o.kt)("h3",{id:"why-is-the-real-time-nature-of-the-recommendation-system-important"},"Why is the real-time nature of the recommendation system important?"),(0,o.kt)("p",null,'Intuitively, when users use personalized news applications, users expect to find articles that match their interests faster; when using short video services, they expect to "flash" content that they are interested in faster; when doing online shopping, I also hope to find the products that I like, faster. All recommendations highlight the word "fast", which is an intuitive manifestation of the "real-time" role of the recommendation system.'),(0,o.kt)("p",null,"From a professional point of view, the real-time performance of the recommendation system is also crucial, which is mainly reflected in the following two aspects:"),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("strong",{parentName:"li"},"The faster the update speed of the recommendation system is, the more it can reflect the user's recent user habits, and the more time-sensitive it can make recommendations to the user.")),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("strong",{parentName:"li"},"The faster the recommendation system is updated, the easier it is for the model to find the latest popular data patterns, and the more it can make the model react to find the latest fashion trends."))),(0,o.kt)("h3",{id:"the-real-time-nature-of-the-feature-of-the-recommendation-system"},'The real-time nature of the "feature" of the recommendation system'),(0,o.kt)("p",null,'Suppose a user has watched a 10-minute "badminton teaching" video in its entirety. Then there is no doubt that the user is interested in the subject of "badminton". The system hopes to continue to recommend "badminton" related videos when the user turns the page next time. However, due to the lack of real-time features of the system, the user\u2019s viewing history cannot be fed back to the recommendation system in real time. As a result, the recommendation system learned that the user had watched the video "Badminton Teaching". It was already half an hour later. Has left the app. This is an example of recommendation failure caused by poor real-time performance of the recommendation system.'),(0,o.kt)("p",null,'It is true that the next time the user opens the application, the recommendation system can use the last user behavior history to recommend "badminton" related videos, but the recommendation system undoubtedly loses what is most likely to increase user viscosity and increase user retention. opportunity.'),(0,o.kt)("h3",{id:"the-real-time-nature-of-the-model-of-the-recommender-system"},'The real-time nature of the "model" of the recommender system'),(0,o.kt)("p",null,'No matter how strong the real-time feature is, the scope of influence is limited to the current user. Compared with the real-time nature of "features", the real-time nature of the recommendation system model is often considered from a more global perspective . The real-time nature of the feature attempts to describe a person with more accurate features, so that the recommendation system can give a recommendation result that is more in line with the person. The real-time nature of the model hopes to capture new data patterns at the global level faster and discover new trends and relevance.'),(0,o.kt)("p",null,"Take, for example, a large number of promotional activities on Double Eleven on an e-commerce website. The real-time nature of the feature will quickly discover the products that the user may be interested in based on the user's recent behavior, but will never find the latest preferences of similar users, the latest correlation information between the products, and the trend information of new activities."),(0,o.kt)("p",null,"To discover such global data changes, the model needs to be updated faster. The most important factor affecting the real-time performance of the model is the training method of the model."),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("strong",{parentName:"li"},"Full update -"),' The most common way of model training is full update.\xa0The model will use all training samples in a certain period of time for retraining, and then replace the "outdated" model with the new trained model. However, the full update requires a large amount of training samples, so the training time required is longer; and the full update is often performed on offline big data platforms, such as spark+tensorflow, so the data delay is also longer, which leads to the full update It is the worst "real-time" model update method. In fact, for a model that has been trained, it is enough to learn only the newly added incremental samples, which is called incremental update.'),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("strong",{parentName:"li"},"Incremental update (Incremental Learning)"),' - Incremental update only feeds newly added samples to the model for incremental learning . Technically, deep learning models often use stochastic gradient descent (SGD) and its variants for learning. The model\'s learning of incremental samples is equivalent to continuing to input incremental samples for gradient descent on the basis of the original samples. Therefore, based on the deep learning model, it is not difficult to change from full update to incremental update. But everything in engineering is a tradeoff, there is never a perfect solution, and incremental updates are no exception. Since only incremental samples are used for learning, the model also converges to the best point of the new sample after multiple epochs, and it is difficult to converge to the global best point of all the original samples + incremental samples. Therefore, in the actual recommendation system, the incremental update and the global update are often combined . After several rounds of incremental update, the global update is performed in a time window with a small business volume, and the model is corrected after the incremental update process. Accumulated errors in. Make trade-offs and trade-offs between "real-time performance" and "global optimization".'),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("strong",{parentName:"li"},"Online learning"),' - "Online learning" is a further improvement of "incremental update", "incremental update" is to perform incremental update when a batch of new samples is obtained, and online learning is to update the model in real time every time a new sample is obtained. Online learning can also be implemented technically through SGD. But if you use the general SGD method, online learning will cause a very serious problem, that is, the sparsity of the model is very poor, opening too many "fragmented" unimportant features. We pay attention to the "sparseness" of the model in a sense that is also an engineering consideration. For example, in a model with an input vector of several million dimensions, if the sparsity of the model is good, the effect of the model can be maintained without affecting the model. , Only make the corresponding weight of the input vector of a very small part of the dimension non-zero, that is to say, when the model is online, the volume of the model is very small, which is undoubtedly beneficial to the entire model serving process. Both the memory space required to store the model and the speed of online inference will benefit from the sparsity of the model. If the SGD method is used to update the model, it is easier to generate a large number of features with small weights than the batch method, which increases the difficulty of model deployment and update. So in order to take into account the training effect and model sparsity in the online learning process, there are a lot of related researches. The most famous ones include Microsoft\'s RDA, Google\'s FOBOS and the most famous FTRL, etc.'),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("strong",{parentName:"li"},"Partial model update")," - Another improvement direction to improve the real-time performance of the model is to perform a partial update of the model. The general idea is to reduce the update frequency of the part with low training efficiency and increase the update frequency of the part with high training efficiency . This approach is representative of the GBDT+LR model of Facebook.")),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"/img/content-blog-raw-blog-real-time-news-personalization-with-flink-untitled.png",src:n(16953).Z})),(0,o.kt)("h2",{id:"data-pipeline-of-a-typical-news-recommendation-system"},"Data pipeline of a typical news recommendation system"),(0,o.kt)("p",null,"When a user is exposed with a list of news articles, a page view events are sent to the backend server and when that user clicks on the news of interest, the action events are also sent to the backend server. After receiving these 2 event streams (page view and clicks), the backend server will send these user behaviour events to the message queue. And message queue finally stores these messages into the distributed file system, such as HDFS."),(0,o.kt)("p",null,"For model training, we need a training sample. The most common sampling technique is negative sampling. In this, we generate 'n' negative samples for each positive event that we receive. Users will only generate behavior for some exposed news samples, which are positive samples, and the remaining exposure samples without behavior are negative samples. After generating positive and negative samples, the model can be trained."),(0,o.kt)("p",null,"The recommendation system with low real-time requirements can use batch processing technology (APACHE spark is a typical tool) to generate samples, as shown in the left figure. Set a timing task, and read the user behavior log and exposure log in the time window from HDFS every other period of time, such as one hour, to perform join operation, generate training samples, and then write the training samples back to HDFS, Then start the training update of the model."),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"/img/content-blog-raw-blog-real-time-news-personalization-with-flink-untitled-1.png",src:n(15691).Z})),(0,o.kt)("h3",{id:"problems"},"Problems"),(0,o.kt)("p",null,"One obvious problem with batch processing is ",(0,o.kt)("strong",{parentName:"p"},"latency"),". The typical cycle of running batch tasks regularly is one hour, which means that there is a delay of at least one hour from sample generation to model training. Sometimes, if the batch platform is overloaded and the tasks need to be queued, the delay will be greater."),(0,o.kt)("p",null,"Another problem is the ",(0,o.kt)("strong",{parentName:"p"},"boundary")," problem. If page view (PV) data is generated at the end of the log time window selected by the batch task, the corresponding action data may fall into the next time window of the batch task, resulting in join failure and false negative samples."),(0,o.kt)("p",null,"A related problem to this is the time synchronization problem. When a news item is exposed to the user, the user may click immediately after the PV data stream is generated, or the user may act after a few minutes, more than ten minutes, or even several hours. This means that after the PV data stream arrives, it needs to wait for a period of time to join with the action data stream. If the waiting time is too long, some samples (positive samples) that should have user behavior will be wrongly marked as negative samples because the user behavior has no time to return. Too long waiting time will damage and increase the system delay. Offline analysis of the delay distribution between the actual action data stream and PV data stream is a very typical exponential distribution."),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"/img/content-blog-raw-blog-real-time-news-personalization-with-flink-untitled-2.png",src:n(24918).Z})),(0,o.kt)("h2",{id:"apache-flink-to-the-rescue"},"Apache Flink to the rescue"),(0,o.kt)("h3",{id:"how-apache-flink-solves-the-latency-problem"},"How Apache Flink solves the latency problem?"),(0,o.kt)("p",null,"In order to enhance the real-time performance, we use Apache Flink framework to rewrite the sample generation logic with stream processing technology. As shown in the right figure above, after the user exposure and behavior logs generated by online services are written into the message queue, instead of waiting for them to drop to HDFS, we directly consume these message flows with Flink. At the same time, Flink reads the necessary feature information from the redis cache and generates the sample message stream directly. The sample message flow is written back to the Kafka queue, and downstream tensorflow can directly consume the message flow for model training."),(0,o.kt)("h3",{id:"how-apache-flink-solved-the-boundary-and-synchronization-problem"},"How Apache Flink solved the boundary and synchronization problem?"),(0,o.kt)("p",null,"As per the exponential distribution (analyzed on a private dataset of a news recommender app), most of the user behavior has reflow within a few minutes. And if few minutes is an acceptable delay, a simple solution is to set a time window with a compromise size. Flink provides window join to implement this logic."),(0,o.kt)("h2",{id:"references"},"References"),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("a",{parentName:"li",href:"https://developpaper.com/flink-streaming-processing-and-real-time-sample-generation-in-recommender-system/"},"https://developpaper.com/flink-streaming-processing-and-real-time-sample-generation-in-recommender-system/")),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("a",{parentName:"li",href:"https://zhuanlan.zhihu.com/p/74813776"},"https://zhuanlan.zhihu.com/p/74813776")),(0,o.kt)("li",{parentName:"ol"},(0,o.kt)("a",{parentName:"li",href:"https://zhuanlan.zhihu.com/p/75597761"},"https://zhuanlan.zhihu.com/p/75597761"))))}p.isMDXComponent=!0},15691:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-real-time-news-personalization-with-flink-untitled-1-319655da20fc7d94210cca6b2a83dbc7.png"},24918:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-real-time-news-personalization-with-flink-untitled-2-55b1c3ab0b3ab90bdefc3cc1a6ec7d86.png"},16953:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-real-time-news-personalization-with-flink-untitled-0a226898c0c217ee517545659bc11b51.png"}}]);