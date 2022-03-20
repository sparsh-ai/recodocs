"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[1203],{3905:function(e,a,t){t.d(a,{Zo:function(){return l},kt:function(){return k}});var n=t(67294);function s(e,a,t){return a in e?Object.defineProperty(e,a,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[a]=t,e}function m(e,a){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);a&&(n=n.filter((function(a){return Object.getOwnPropertyDescriptor(e,a).enumerable}))),t.push.apply(t,n)}return t}function r(e){for(var a=1;a<arguments.length;a++){var t=null!=arguments[a]?arguments[a]:{};a%2?m(Object(t),!0).forEach((function(a){s(e,a,t[a])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):m(Object(t)).forEach((function(a){Object.defineProperty(e,a,Object.getOwnPropertyDescriptor(t,a))}))}return e}function p(e,a){if(null==e)return{};var t,n,s=function(e,a){if(null==e)return{};var t,n,s={},m=Object.keys(e);for(n=0;n<m.length;n++)t=m[n],a.indexOf(t)>=0||(s[t]=e[t]);return s}(e,a);if(Object.getOwnPropertySymbols){var m=Object.getOwnPropertySymbols(e);for(n=0;n<m.length;n++)t=m[n],a.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(s[t]=e[t])}return s}var i=n.createContext({}),o=function(e){var a=n.useContext(i),t=a;return e&&(t="function"==typeof e?e(a):r(r({},a),e)),t},l=function(e){var a=o(e.components);return n.createElement(i.Provider,{value:a},e.children)},c={inlineCode:"code",wrapper:function(e){var a=e.children;return n.createElement(n.Fragment,{},a)}},N=n.forwardRef((function(e,a){var t=e.components,s=e.mdxType,m=e.originalType,i=e.parentName,l=p(e,["components","mdxType","originalType","parentName"]),N=o(t),k=s,d=N["".concat(i,".").concat(k)]||N[k]||c[k]||m;return t?n.createElement(d,r(r({ref:a},l),{},{components:t})):n.createElement(d,r({ref:a},l))}));function k(e,a){var t=arguments,s=a&&a.mdxType;if("string"==typeof e||s){var m=t.length,r=new Array(m);r[0]=N;var p={};for(var i in a)hasOwnProperty.call(a,i)&&(p[i]=a[i]);p.originalType=e,p.mdxType="string"==typeof e?e:s,r[1]=p;for(var o=2;o<m;o++)r[o]=t[o];return n.createElement.apply(null,r)}return n.createElement.apply(null,t)}N.displayName="MDXCreateElement"},19860:function(e,a,t){t.r(a),t.d(a,{frontMatter:function(){return p},contentTitle:function(){return i},metadata:function(){return o},toc:function(){return l},default:function(){return N}});var n=t(87462),s=t(63366),m=(t(67294),t(3905)),r=["components"],p={},i="Word2vec",o={unversionedId:"models/word2vec",id:"models/word2vec",title:"Word2vec",description:"/img/content-models-raw-mp2-word2vec-untitled.png",source:"@site/docs/models/word2vec.md",sourceDirName:"models",slug:"/models/word2vec",permalink:"/docs/models/word2vec",editUrl:"https://github.com/sparsh-ai/ml-utils/docs/models/word2vec.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Wide and Deep",permalink:"/docs/models/wide-and-deep"},next:{title:"xDeepFM",permalink:"/docs/models/xdeepfm"}},l=[{value:"<strong>CBOW and Skip-Gram</strong>",id:"cbow-and-skip-gram",children:[],level:3},{value:"Skip-gram model",id:"skip-gram-model",children:[],level:2},{value:"Links",id:"links",children:[],level:2}],c={toc:l};function N(e){var a=e.components,p=(0,s.Z)(e,r);return(0,m.kt)("wrapper",(0,n.Z)({},c,p,{components:a,mdxType:"MDXLayout"}),(0,m.kt)("h1",{id:"word2vec"},"Word2vec"),(0,m.kt)("p",null,(0,m.kt)("img",{alt:"/img/content-models-raw-mp2-word2vec-untitled.png",src:t(79783).Z})),(0,m.kt)("p",null,"Word2vec uses the co-occurrence of words in a sentence to learn embeddings for each word that capture the semantic meaning of that word. At its core, word2vec is a simple, shallow neural network, with a single hidden layer. In its skip-gram version, it takes as input a word, and tries to predict the context of words around it as the output. For instance, consider this sentence:"),(0,m.kt)("p",null,"\u201cThe cat\xa0",(0,m.kt)("em",{parentName:"p"},"jumped"),"\xa0over the puddle.\u201d"),(0,m.kt)("p",null,"Given the central word \u201cjumped,\u201d the model will be able to predict the surrounding words: \u201cThe,\u201d \u201ccat,\u201d \u201cover,\u201d \u201cthe,\u201d \u201cpuddle.\u201d"),(0,m.kt)("p",null,"Another approach\u2014Continuous Bag of Words (CBOW)\u2014treats the words \u201cThe,\u201d \u201ccat,\u201d \u201cover,\u201d \u201cthe,\u201d and \u201cpuddle\u201d as the context, and predicts the center word: \u201cjumped.\u201d"),(0,m.kt)("h3",{id:"cbow-and-skip-gram"},(0,m.kt)("strong",{parentName:"h3"},"CBOW and Skip-Gram")),(0,m.kt)("p",null,"There are two models for Word2Vec:\xa0Continous Bag Of Words (CBOW)\xa0and\xa0Skip-Gram. While Skip-Gram model predicts context words given a center word, CBOW model predicts a center word given context words. According to Mikolov:"),(0,m.kt)("p",null,"Skip-gram: works well with small amount of the training data, represents well even rare words or phrases"),(0,m.kt)("p",null,"CBOW: several times faster to train than the skip-gram, slightly better accuracy for the frequent words"),(0,m.kt)("p",null,"Skip-Gram model is a better choice most of the time due to its ability to predict infrequent words, but this comes at the price of increased computational cost. If training time is a big concern, and you have large enough data to overcome the issue of predicting infrequent words, CBOW model may be a more viable choice. The details of CBOW model won't be covered in this post."),(0,m.kt)("h2",{id:"skip-gram-model"},"Skip-gram model"),(0,m.kt)("p",null,"Skip-Gram model seeks to optimize the word weight (embedding) matrix by correctly predicting context words, given a center word. In the other words, the model wants to maximize the probability of correctly predicting all context words at the same time, given a center word. Maximizing the probability of predicting context words leads to optimizing the weight matrix (\u03b8) that best represents words in a vector space. Mathematically, it can be expressed as: ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mi",{parentName:"mrow"},(0,m.kt)("munder",{parentName:"mi"},(0,m.kt)("mo",{parentName:"munder"},(0,m.kt)("mtext",{parentName:"mo"},"argmax")),(0,m.kt)("mi",{parentName:"munder"},"\u03b8"))),(0,m.kt)("mtext",{parentName:"mrow"},"\u2009"),(0,m.kt)("mtext",{parentName:"mrow"},"\u2009"),(0,m.kt)("mi",{parentName:"mrow"},"l"),(0,m.kt)("mi",{parentName:"mrow"},"o"),(0,m.kt)("mi",{parentName:"mrow"},"g"),(0,m.kt)("mtext",{parentName:"mrow"},"\xa0"),(0,m.kt)("mi",{parentName:"mrow"},"p"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},"("),(0,m.kt)("msub",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msub"},"w"),(0,m.kt)("mn",{parentName:"msub"},"1")),(0,m.kt)("mo",{parentName:"mrow",separator:"true"},","),(0,m.kt)("msub",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msub"},"w"),(0,m.kt)("mn",{parentName:"msub"},"2")),(0,m.kt)("mo",{parentName:"mrow",separator:"true"},","),(0,m.kt)("mi",{parentName:"mrow",mathvariant:"normal"},"."),(0,m.kt)("mi",{parentName:"mrow",mathvariant:"normal"},"."),(0,m.kt)("mi",{parentName:"mrow",mathvariant:"normal"},"."),(0,m.kt)("mo",{parentName:"mrow",separator:"true"},","),(0,m.kt)("msub",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msub"},"w"),(0,m.kt)("mi",{parentName:"msub"},"C")),(0,m.kt)("mi",{parentName:"mrow",mathvariant:"normal"},"\u2223"),(0,m.kt)("msub",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msub"},"w"),(0,m.kt)("mrow",{parentName:"msub"},(0,m.kt)("mi",{parentName:"mrow"},"c"),(0,m.kt)("mi",{parentName:"mrow"},"e"),(0,m.kt)("mi",{parentName:"mrow"},"n"),(0,m.kt)("mi",{parentName:"mrow"},"t"),(0,m.kt)("mi",{parentName:"mrow"},"e"),(0,m.kt)("mi",{parentName:"mrow"},"r"))),(0,m.kt)("mo",{parentName:"mrow",separator:"true"},";"),(0,m.kt)("mtext",{parentName:"mrow"},"\u2009"),(0,m.kt)("mi",{parentName:"mrow"},"\u03b8"),(0,m.kt)("mo",{parentName:"mrow",stretchy:"false"},")")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\\underset{\\theta}{\\text{argmax}} \\,\\, log\\ p(w_{1}, w_{2}, ... , w_{C}|w_{center}; \\, \\theta)")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"1.6965em",verticalAlign:"-0.9465em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mop op-limits"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.4306em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.1535em",marginLeft:"0em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight",style:{marginRight:"0.02778em"}},"\u03b8")))),(0,m.kt)("span",{parentName:"span",style:{top:"-3em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3em"}}),(0,m.kt)("span",{parentName:"span"},(0,m.kt)("span",{parentName:"span",className:"mop"},(0,m.kt)("span",{parentName:"span",className:"mord text"},(0,m.kt)("span",{parentName:"span",className:"mord"},"argmax")))))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.9465em"}},(0,m.kt)("span",{parentName:"span"})))))),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.01968em"}},"l"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"o"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"g"),(0,m.kt)("span",{parentName:"span",className:"mspace"},"\xa0"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"p"),(0,m.kt)("span",{parentName:"span",className:"mopen"},"("),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.02691em"}},"w"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.3011em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.55em",marginLeft:"-0.0269em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},"1"))))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,m.kt)("span",{parentName:"span"})))))),(0,m.kt)("span",{parentName:"span",className:"mpunct"},","),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.02691em"}},"w"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.3011em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.55em",marginLeft:"-0.0269em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},"2"))))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,m.kt)("span",{parentName:"span"})))))),(0,m.kt)("span",{parentName:"span",className:"mpunct"},","),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},"..."),(0,m.kt)("span",{parentName:"span",className:"mpunct"},","),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.02691em"}},"w"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.3283em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.55em",marginLeft:"-0.0269em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight",style:{marginRight:"0.07153em"}},"C"))))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,m.kt)("span",{parentName:"span"})))))),(0,m.kt)("span",{parentName:"span",className:"mord"},"\u2223"),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.02691em"}},"w"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.2806em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.55em",marginLeft:"-0.0269em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"ce"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"n"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight"},"t"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal mtight",style:{marginRight:"0.02778em"}},"er"))))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,m.kt)("span",{parentName:"span"})))))),(0,m.kt)("span",{parentName:"span",className:"mpunct"},";"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.02778em"}},"\u03b8"),(0,m.kt)("span",{parentName:"span",className:"mclose"},")")))))),(0,m.kt)("h2",{id:"links"},"Links"),(0,m.kt)("ol",null,(0,m.kt)("li",{parentName:"ol"},(0,m.kt)("a",{parentName:"li",href:"https://aegis4048.github.io/demystifying_neural_network_in_skip_gram_language_modeling"},"https://aegis4048.github.io/demystifying_neural_network_in_skip_gram_language_modeling")," ",(0,m.kt)("inlineCode",{parentName:"li"},"blog")),(0,m.kt)("li",{parentName:"ol"},(0,m.kt)("a",{parentName:"li",href:"https://aegis4048.github.io/optimize_computational_efficiency_of_skip-gram_with_negative_sampling"},"https://aegis4048.github.io/optimize_computational_efficiency_of_skip-gram_with_negative_sampling")," ",(0,m.kt)("inlineCode",{parentName:"li"},"blog"))))}N.isMDXComponent=!0},79783:function(e,a,t){a.Z=t.p+"assets/images/content-models-raw-mp2-word2vec-untitled-960b0c9034e1278d0dc8eeadf507f5fb.png"}}]);