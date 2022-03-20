"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[5292],{3905:function(e,t,r){r.d(t,{Zo:function(){return u},kt:function(){return d}});var n=r(67294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function i(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function o(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?i(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function s(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},i=Object.keys(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var c=n.createContext({}),l=function(e){var t=n.useContext(c),r=t;return e&&(r="function"==typeof e?e(t):o(o({},t),e)),r},u=function(e){var t=l(e.components);return n.createElement(c.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},f=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,i=e.originalType,c=e.parentName,u=s(e,["components","mdxType","originalType","parentName"]),f=l(r),d=a,m=f["".concat(c,".").concat(d)]||f[d]||p[d]||i;return r?n.createElement(m,o(o({ref:t},u),{},{components:r})):n.createElement(m,o({ref:t},u))}));function d(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=r.length,o=new Array(i);o[0]=f;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:a,o[1]=s;for(var l=2;l<i;l++)o[l]=r[l];return n.createElement.apply(null,o)}return n.createElement.apply(null,r)}f.displayName="MDXCreateElement"},7691:function(e,t,r){r.r(t),r.d(t,{frontMatter:function(){return s},contentTitle:function(){return c},metadata:function(){return l},toc:function(){return u},default:function(){return f}});var n=r(87462),a=r(63366),i=(r(67294),r(3905)),o=["components"],s={},c="Netflix Personalize Images",l={unversionedId:"concept-extras/success-stories/netflix-personalize-images",id:"concept-extras/success-stories/netflix-personalize-images",title:"Netflix Personalize Images",description:"Read here on Netflix's blog",source:"@site/docs/concept-extras/success-stories/netflix-personalize-images.md",sourceDirName:"concept-extras/success-stories",slug:"/concept-extras/success-stories/netflix-personalize-images",permalink:"/docs/concept-extras/success-stories/netflix-personalize-images",editUrl:"https://github.com/sparsh-ai/ml-utils/docs/concept-extras/success-stories/netflix-personalize-images.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"MarketCloud Real-time",permalink:"/docs/concept-extras/success-stories/marketcloud-real-time"},next:{title:"Pinterest Multi-task Learning",permalink:"/docs/concept-extras/success-stories/pinterest-multi-task-learning"}},u=[],p={toc:u};function f(e){var t=e.components,s=(0,a.Z)(e,o);return(0,i.kt)("wrapper",(0,n.Z)({},p,s,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"netflix-personalize-images"},"Netflix Personalize Images"),(0,i.kt)("p",null,(0,i.kt)("a",{parentName:"p",href:"https://netflixtechblog.com/artwork-personalization-c589f074ad76"},"Read here on Netflix's blog")),(0,i.kt)("p",null,"The bandit can choose from a set of images for each show (i.e., action) and observe the number of minutes the user played the show after being impressed with the image (i.e., reward). It also has information about user attributes (e.g., titles played, genres played, country, language preferences), day of the week, time of day, etc. (i.e., context)."),(0,i.kt)("p",null,"For offline evaluation of the bandit, they apply replay on the bandit\u2019s predicted image and the random image shown during the exploration phase. They first get the bandit\u2019s predicted image for each user-show pair. Then, they try to match it with the random images shown to users in the exploration phase. If the predicted image matches the randomly assigned image, that predicted-random match can be used for evaluation."),(0,i.kt)("p",null,(0,i.kt)("img",{alt:"/img/content-concepts-case-studies-raw-case-studies-netflix-personalize-images-untitled.png",src:r(93128).Z})),(0,i.kt)("p",null,"From the set of predicted-random matches, they check if the user played the title or not. The main metric of interest is the number of quality plays over the number of impressions (i.e., take fraction)\u2014for the n images that were recommended, how many resulted in the user watching the show?"),(0,i.kt)("p",null,"The benefit of replay is that it\u2019s an unbiased metric when accounting for the probability of each image shown during exploration. Having the probability allows us to weigh the reward to control for bias in image display rates, either in exploration or production. (Also see this ",(0,i.kt)("a",{parentName:"p",href:"http://www.cs.cornell.edu/~adith/CfactSIGIR2016/"},"SIGIR tutorial on counterfactual evaluation"),") The downside is that it requires a lot of data, and there could be high variance in evaluation metrics if there are few matches between the predicted and random data. Nonetheless, techniques such as ",(0,i.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/1103.4601"},"doubly robust estimation")," can help."))}f.isMDXComponent=!0},93128:function(e,t,r){t.Z=r.p+"assets/images/content-concepts-case-studies-raw-case-studies-netflix-personalize-images-untitled-d05d0e60d67165afb049a6b123a88ad5.png"}}]);