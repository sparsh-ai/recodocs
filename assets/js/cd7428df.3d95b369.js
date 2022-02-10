"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[6640],{3905:function(e,t,n){n.d(t,{Zo:function(){return l},kt:function(){return p}});var r=n(67294);function s(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function c(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){s(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function m(e,t){if(null==e)return{};var n,r,s=function(e,t){if(null==e)return{};var n,r,s={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(s[n]=e[n]);return s}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(s[n]=e[n])}return s}var i=r.createContext({}),a=function(e){var t=r.useContext(i),n=t;return e&&(n="function"==typeof e?e(t):c(c({},t),e)),n},l=function(e){var t=a(e.components);return r.createElement(i.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},u=r.forwardRef((function(e,t){var n=e.components,s=e.mdxType,o=e.originalType,i=e.parentName,l=m(e,["components","mdxType","originalType","parentName"]),u=a(n),p=s,y=u["".concat(i,".").concat(p)]||u[p]||d[p]||o;return n?r.createElement(y,c(c({ref:t},l),{},{components:n})):r.createElement(y,c({ref:t},l))}));function p(e,t){var n=arguments,s=t&&t.mdxType;if("string"==typeof e||s){var o=n.length,c=new Array(o);c[0]=u;var m={};for(var i in t)hasOwnProperty.call(t,i)&&(m[i]=t[i]);m.originalType=e,m.mdxType="string"==typeof e?e:s,c[1]=m;for(var a=2;a<o;a++)c[a]=n[a];return r.createElement.apply(null,c)}return r.createElement.apply(null,n)}u.displayName="MDXCreateElement"},46255:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return m},contentTitle:function(){return i},metadata:function(){return a},toc:function(){return l},default:function(){return u}});var r=n(87462),s=n(63366),o=(n(67294),n(3905)),c=["components"],m={},i="Types of Recommender Systems",a={unversionedId:"concept-basics/types-of-recommender-systems",id:"concept-basics/types-of-recommender-systems",title:"Types of Recommender Systems",description:"Group Recommender System",source:"@site/docs/concept-basics/types-of-recommender-systems.md",sourceDirName:"concept-basics",slug:"/concept-basics/types-of-recommender-systems",permalink:"/docs/concept-basics/types-of-recommender-systems",editUrl:"https://github.com/recohut/docs/docs/docs/concept-basics/types-of-recommender-systems.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Tasks",permalink:"/docs/concept-basics/tasks"},next:{title:"Concept - Extras",permalink:"/docs/concept-extras/"}},l=[{value:"Group Recommender System",id:"group-recommender-system",children:[],level:2},{value:"Multi-Stakeholder Recommender System",id:"multi-stakeholder-recommender-system",children:[],level:2},{value:"Multi-Task Recommender System",id:"multi-task-recommender-system",children:[],level:2},{value:"Content-based Recommender System",id:"content-based-recommender-system",children:[],level:2},{value:"Collaborative Filtering Recommender System",id:"collaborative-filtering-recommender-system",children:[],level:2},{value:"Session-based Recommender System",id:"session-based-recommender-system",children:[],level:2},{value:"Context-aware Recommender System",id:"context-aware-recommender-system",children:[],level:2},{value:"Hybrid Recommender System",id:"hybrid-recommender-system",children:[],level:2}],d={toc:l};function u(e){var t=e.components,m=(0,s.Z)(e,c);return(0,o.kt)("wrapper",(0,r.Z)({},d,m,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"types-of-recommender-systems"},"Types of Recommender Systems"),(0,o.kt)("h2",{id:"group-recommender-system"},"Group Recommender System"),(0,o.kt)("p",null,"Recommend items to a group of users, e.g., group dining"),(0,o.kt)("h2",{id:"multi-stakeholder-recommender-system"},"Multi-Stakeholder Recommender System"),(0,o.kt)("p",null,"Produce recommendations by considering multiple stakeholders, e.g., buyers and sellers on eBay"),(0,o.kt)("h2",{id:"multi-task-recommender-system"},"Multi-Task Recommender System"),(0,o.kt)("p",null,"Build joint learning model by considering multiple tasks, e.g., RecSys + opinion texts"),(0,o.kt)("h2",{id:"content-based-recommender-system"},"Content-based Recommender System"),(0,o.kt)("p",null,"Below figure shows the high-level architecture of a CBRS, one of many possible architectures."),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"Untitled",src:n(98632).Z})),(0,o.kt)("h2",{id:"collaborative-filtering-recommender-system"},"Collaborative Filtering Recommender System"),(0,o.kt)("p",null,"A graph-powered collaborative filtering recommender system"),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"Untitled",src:n(75553).Z})),(0,o.kt)("h2",{id:"session-based-recommender-system"},"Session-based Recommender System"),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"Untitled",src:n(66489).Z})),(0,o.kt)("h2",{id:"context-aware-recommender-system"},"Context-aware Recommender System"),(0,o.kt)("p",null,"Incorporate context info (time, location, etc) into RecSys"),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"Untitled",src:n(48214).Z})),(0,o.kt)("p",null,"Based on how the contextual information, the current user, and the current item are used during the recommendation process, context-aware recommendation systems can take one of the three forms shown below:"),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"Untitled",src:n(59855).Z})),(0,o.kt)("h2",{id:"hybrid-recommender-system"},"Hybrid Recommender System"),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"Untitled",src:n(64124).Z})),(0,o.kt)("p",null,"Hybridization design techniques"),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"Untitled",src:n(89008).Z})))}u.isMDXComponent=!0},75553:function(e,t,n){t.Z=n.p+"assets/images/content-concepts-raw-types-of-recommender-systems-untitled-1-38fec0818f6f9f2f6e820ff0835d5738.png"},66489:function(e,t,n){t.Z=n.p+"assets/images/content-concepts-raw-types-of-recommender-systems-untitled-2-e73a604b27522a7aafad05abdc11dd8d.png"},48214:function(e,t,n){t.Z=n.p+"assets/images/content-concepts-raw-types-of-recommender-systems-untitled-3-af24b5c345caad7ca45cda3b02c292a6.png"},59855:function(e,t,n){t.Z=n.p+"assets/images/content-concepts-raw-types-of-recommender-systems-untitled-4-0353256353337fe812e49b17cc9dc694.png"},64124:function(e,t,n){t.Z=n.p+"assets/images/content-concepts-raw-types-of-recommender-systems-untitled-5-8e15438be50ee446889215768155a8a7.png"},89008:function(e,t,n){t.Z=n.p+"assets/images/content-concepts-raw-types-of-recommender-systems-untitled-6-c5a1f18730362df43215136c1dff9fcb.png"},98632:function(e,t,n){t.Z=n.p+"assets/images/content-concepts-raw-types-of-recommender-systems-untitled-0839380b194dae435212787f7169c307.png"}}]);