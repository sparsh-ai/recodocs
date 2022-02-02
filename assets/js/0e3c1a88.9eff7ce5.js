"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[4816],{3905:function(e,t,n){n.d(t,{Zo:function(){return m},kt:function(){return p}});var r=n(67294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var s=r.createContext({}),c=function(e){var t=r.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},m=function(e){var t=c(e.components);return r.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,i=e.originalType,s=e.parentName,m=l(e,["components","mdxType","originalType","parentName"]),d=c(n),p=o,b=d["".concat(s,".").concat(p)]||d[p]||u[p]||i;return n?r.createElement(b,a(a({ref:t},m),{},{components:n})):r.createElement(b,a({ref:t},m))}));function p(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var i=n.length,a=new Array(i);a[0]=d;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:o,a[1]=l;for(var c=2;c<i;c++)a[c]=n[c];return r.createElement.apply(null,a)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},37161:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return l},contentTitle:function(){return s},metadata:function(){return c},assets:function(){return m},toc:function(){return u},default:function(){return p}});var r=n(87462),o=n(63366),i=(n(67294),n(3905)),a=["components"],l={title:"Document Recommendation",authors:"sparsh",tags:["nlp","similarity"]},s=void 0,c={permalink:"/blog/2021/10/01/document-recommendation",editUrl:"https://github.com/recohut/docs/blog/blog/2021-10-01-document-recommendation.mdx",source:"@site/blog/2021-10-01-document-recommendation.mdx",title:"Document Recommendation",description:"/img/content-blog-raw-blog-document-recommendation-untitled.png",date:"2021-10-01T00:00:00.000Z",formattedDate:"October 1, 2021",tags:[{label:"nlp",permalink:"/blog/tags/nlp"},{label:"similarity",permalink:"/blog/tags/similarity"}],readingTime:1.285,truncated:!1,authors:[{name:"Sparsh Agarwal",title:"Principal Developer",url:"https://github.com/sparsh-ai",imageURL:"https://avatars.githubusercontent.com/u/62965911?v=4",key:"sparsh"}],prevItem:{title:"Distributed Training of Recommender Systems",permalink:"/blog/2021/10/01/distributed-training-of-recommender-systems"},nextItem:{title:"Fake Voice Detection",permalink:"/blog/2021/10/01/fake-voice-detection"}},m={authorsImageUrls:[void 0]},u=[{value:"<strong>Introduction</strong>",id:"introduction",children:[{value:"Business objective",id:"business-objective",children:[],level:3},{value:"Technical objective",id:"technical-objective",children:[],level:3}],level:2},{value:"<strong>Proposed Framework 1 \u2014 Hybrid Recommender System</strong>",id:"proposed-framework-1--hybrid-recommender-system",children:[],level:2},{value:"<strong>Proposed Framework 2 \u2014 Content-based Recommender System</strong>",id:"proposed-framework-2--content-based-recommender-system",children:[],level:2},{value:"<strong>Results and Discussion</strong>",id:"results-and-discussion",children:[{value:"<strong>Code</strong>",id:"code",children:[],level:3}],level:2}],d={toc:u};function p(e){var t=e.components,l=(0,o.Z)(e,a);return(0,i.kt)("wrapper",(0,r.Z)({},d,l,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("p",null,(0,i.kt)("img",{alt:"/img/content-blog-raw-blog-document-recommendation-untitled.png",src:n(14143).Z})),(0,i.kt)("h2",{id:"introduction"},(0,i.kt)("strong",{parentName:"h2"},"Introduction")),(0,i.kt)("h3",{id:"business-objective"},"Business objective"),(0,i.kt)("p",null,"For the given user query, recommend relevant documents (BRM_ifam)"),(0,i.kt)("h3",{id:"technical-objective"},"Technical objective"),(0,i.kt)("p",null,"1-to-N mapping of given input text"),(0,i.kt)("h2",{id:"proposed-framework-1--hybrid-recommender-system"},(0,i.kt)("strong",{parentName:"h2"},"Proposed Framework 1 \u2014 Hybrid Recommender System")),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"Text \u2192 Vector (Universal Sentence Embedding with TF Hub)"),(0,i.kt)("li",{parentName:"ul"},"Vector \u2192 Content-based Filtering Recommendation"),(0,i.kt)("li",{parentName:"ul"},"Index \u2192 Interaction Matrix"),(0,i.kt)("li",{parentName:"ul"},"Interaction Matrix \u2192 Collaborative Filtering Recommendation"),(0,i.kt)("li",{parentName:"ul"},"Collaborative + Content-based \u2192 Hybrid Recommendation"),(0,i.kt)("li",{parentName:"ul"},"Evaluation: Area-under-curve")),(0,i.kt)("h2",{id:"proposed-framework-2--content-based-recommender-system"},(0,i.kt)("strong",{parentName:"h2"},"Proposed Framework 2 \u2014 Content-based Recommender System")),(0,i.kt)("ol",null,(0,i.kt)("li",{parentName:"ol"},"Find A most similar user \u2192 Cosine similarity"),(0,i.kt)("li",{parentName:"ol"},"For each user in A, find TopK Most Similar Items \u2192 Map Argsort"),(0,i.kt)("li",{parentName:"ol"},"For each item Find TopL Most Similar Items \u2192 Cosine similarity"),(0,i.kt)("li",{parentName:"ol"},"Display"),(0,i.kt)("li",{parentName:"ol"},"Implement an evaluation metric"),(0,i.kt)("li",{parentName:"ol"},"Evaluate")),(0,i.kt)("h2",{id:"results-and-discussion"},(0,i.kt)("strong",{parentName:"h2"},"Results and Discussion")),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"build.py \u2192 this script will take the training data as input and save all the required files in the same working directory"),(0,i.kt)("li",{parentName:"ul"},"recommend.py \u2192 this script will take the user query as input and predict top-K BRM recommendations")),(0,i.kt)("p",null,"Variables (during recommendation, you will be asked 2\u20133 choices, the meaning of those choices are as following)"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"top-K \u2014 how many top items you want to get in recommendation"),(0,i.kt)("li",{parentName:"ul"},"secondary items: this will determine how many similar items you would like to add in consideration, for each primary matching item"),(0,i.kt)("li",{parentName:"ul"},"sorted by frequency: since multiple input queries might point to same output, therefore this option allows to take that frequence count of outputs in consideration and will move the more frequent items at the top.")),(0,i.kt)("h3",{id:"code"},(0,i.kt)("strong",{parentName:"h3"},"Code")),(0,i.kt)("p",null,(0,i.kt)("a",{parentName:"p",href:"https://gist.github.com/sparsh-ai/4e5f06ba3c55192b33a276ee67dbd42c#file-text-recommendations-ipynb"},"https://gist.github.com/sparsh-ai/4e5f06ba3c55192b33a276ee67dbd42c#file-text-recommendations-ipynb")))}p.isMDXComponent=!0},14143:function(e,t,n){t.Z=n.p+"assets/images/content-blog-raw-blog-document-recommendation-untitled-ccbcca01bf06db66ebeebcfe4da46778.png"}}]);