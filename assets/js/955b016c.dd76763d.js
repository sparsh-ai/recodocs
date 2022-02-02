"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[898],{3905:function(e,t,n){n.d(t,{Zo:function(){return d},kt:function(){return u}});var r=n(67294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var l=r.createContext({}),c=function(e){var t=r.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},d=function(e){var t=c(e.components);return r.createElement(l.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,a=e.originalType,l=e.parentName,d=s(e,["components","mdxType","originalType","parentName"]),m=c(n),u=o,h=m["".concat(l,".").concat(u)]||m[u]||p[u]||a;return n?r.createElement(h,i(i({ref:t},d),{},{components:n})):r.createElement(h,i({ref:t},d))}));function u(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=n.length,i=new Array(a);i[0]=m;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s.mdxType="string"==typeof e?e:o,i[1]=s;for(var c=2;c<a;c++)i[c]=n[c];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},20865:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return s},contentTitle:function(){return l},metadata:function(){return c},toc:function(){return d},default:function(){return m}});var r=n(87462),o=n(63366),a=(n(67294),n(3905)),i=["components"],s={},l="GCE-GNN",c={unversionedId:"models/gce-gnn",id:"models/gce-gnn",title:"GCE-GNN",description:"GCE-GNN stands for Global Context Enhanced Graph Neural Networks. It exploit item transitions over all sessions in a more subtle manner for better inferring the user preference of the current session.",source:"@site/docs/models/gce-gnn.md",sourceDirName:"models",slug:"/models/gce-gnn",permalink:"/docs/models/gce-gnn",editUrl:"https://github.com/recohut/docs/docs/docs/models/gce-gnn.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"GC-SAN",permalink:"/docs/models/gc-san"},next:{title:"GRU4Rec",permalink:"/docs/models/gru4rec"}},d=[{value:"Architecture",id:"architecture",children:[],level:2}],p={toc:d};function m(e){var t=e.components,s=(0,o.Z)(e,i);return(0,a.kt)("wrapper",(0,r.Z)({},p,s,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"gce-gnn"},"GCE-GNN"),(0,a.kt)("p",null,"GCE-GNN stands for Global Context Enhanced Graph Neural Networks. It exploit item transitions over all sessions in a more subtle manner for better inferring the user preference of the current session."),(0,a.kt)("div",{className:"admonition admonition-info alert alert--info"},(0,a.kt)("div",{parentName:"div",className:"admonition-heading"},(0,a.kt)("h5",{parentName:"div"},(0,a.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,a.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,a.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"}))),"research paper")),(0,a.kt)("div",{parentName:"div",className:"admonition-content"},(0,a.kt)("p",{parentName:"div"},(0,a.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2106.05081"},"Zhang et. al., \u201c",(0,a.kt)("em",{parentName:"a"},"Global Context Enhanced Graph Neural Networks for Session-based Recommendation"),"\u201d. arXiv, 2021.")),(0,a.kt)("blockquote",{parentName:"div"},(0,a.kt)("p",{parentName:"blockquote"},"Session-based recommendation (SBR) is a challenging task, which aims at recommending items based on anonymous behavior sequences. Almost all the existing solutions for SBR model user preference only based on the current session without exploiting the other sessions, which may contain both relevant and irrelevant item-transitions to the current session. This paper proposes a novel approach, called Global Context Enhanced Graph Neural Networks (GCE-GNN) to exploit item transitions over all sessions in a more subtle manner for better inferring the user preference of the current session. Specifically, GCE-GNN learns two levels of item embeddings from session graph and global graph, respectively: (i) Session graph, which is to learn the session-level item embedding by modeling pairwise item-transitions within the current session; and (ii) Global graph, which is to learn the global-level item embedding by modeling pairwise item-transitions over all sessions. In GCE-GNN, we propose a novel global-level item representation learning layer, which employs a session-aware attention mechanism to recursively incorporate the neighbors' embeddings of each node on the global graph. We also design a session-level item representation learning layer, which employs a GNN on the session graph to learn session-level item embeddings within the current session. Moreover, GCE-GNN aggregates the learnt item representations in the two levels with a soft attention mechanism. Experiments on three benchmark datasets demonstrate that GCE-GNN outperforms the state-of-the-art methods consistently.")))),(0,a.kt)("h2",{id:"architecture"},"Architecture"),(0,a.kt)("p",null,(0,a.kt)("img",{alt:"An overview of the proposed framework. Firstly, a global graph is constructed based on all training session sequences. Then for each session, a global feature encoder and local feature encoder will be used to extract node feature with global context and local context. Then the model incorporates position information to learn the contribution of each item to the next predicted item. Finally, candidate items will be scored.",src:n(89069).Z})),(0,a.kt)("p",null,"An overview of the proposed framework. Firstly, a global graph is constructed based on all training session sequences. Then for each session, a global feature encoder and local feature encoder will be used to extract node feature with global context and local context. Then the model incorporates position information to learn the contribution of each item to the next predicted item. Finally, candidate items will be scored."))}m.isMDXComponent=!0},89069:function(e,t,n){t.Z=n.p+"assets/images/content-models-raw-mp1-gce-gnn-untitled-22ec10bb9dcfaa871ddbab561c5b240b.png"}}]);