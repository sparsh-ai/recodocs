"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[7082],{3905:function(e,t,n){n.d(t,{Zo:function(){return u},kt:function(){return d}});var r=n(67294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function s(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var c=r.createContext({}),l=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):s(s({},t),e)),n},u=function(e){var t=l(e.components);return r.createElement(c.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,i=e.originalType,c=e.parentName,u=o(e,["components","mdxType","originalType","parentName"]),m=l(n),d=a,h=m["".concat(c,".").concat(d)]||m[d]||p[d]||i;return n?r.createElement(h,s(s({ref:t},u),{},{components:n})):r.createElement(h,s({ref:t},u))}));function d(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=n.length,s=new Array(i);s[0]=m;var o={};for(var c in t)hasOwnProperty.call(t,c)&&(o[c]=t[c]);o.originalType=e,o.mdxType="string"==typeof e?e:a,s[1]=o;for(var l=2;l<i;l++)s[l]=n[l];return r.createElement.apply(null,s)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},20370:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return o},contentTitle:function(){return c},metadata:function(){return l},toc:function(){return u},default:function(){return m}});var r=n(87462),a=n(63366),i=(n(67294),n(3905)),s=["components"],o={},c="Challenges",l={unversionedId:"concept-basics/challenges",id:"concept-basics/challenges",title:"Challenges",description:"The construction of effective Recommender Systems (RS) is a complex process, mainly due to the nature of RSs which involves large scale software-systems and human interactions. Iterative development processes require deep understanding of a current baseline as well as the ability to estimate the impact of changes in multiple variables of interest. Simulations are well suited to address both challenges and potentially leading to a high velocity construction process, a fundamental requirement in commercial contexts. Recently, there has been significant interest in RS Simulation Platforms, which allow RS developers to easily craft simulated environments where their systems can be analyzed.",source:"@site/docs/concept-basics/challenges.md",sourceDirName:"concept-basics",slug:"/concept-basics/challenges",permalink:"/docs/concept-basics/challenges",editUrl:"https://github.com/recohut/docs/docs/docs/concept-basics/challenges.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Introduction",permalink:"/docs/intro"},next:{title:"Session-based Recommenders",permalink:"/docs/concept-basics/session-based-recommenders"}},u=[],p={toc:u};function m(e){var t=e.components,n=(0,a.Z)(e,s);return(0,i.kt)("wrapper",(0,r.Z)({},p,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"challenges"},"Challenges"),(0,i.kt)("p",null,"The construction of effective Recommender Systems (RS) is a complex process, mainly due to the nature of RSs which involves large scale software-systems and human interactions. Iterative development processes require deep understanding of a current baseline as well as the ability to estimate the impact of changes in multiple variables of interest. Simulations are well suited to address both challenges and potentially leading to a high velocity construction process, a fundamental requirement in commercial contexts. Recently, there has been significant interest in RS Simulation Platforms, which allow RS developers to easily craft simulated environments where their systems can be analyzed."),(0,i.kt)("p",null,"One of the most challenging aspects in estimating the causal impact of a new version on all the relevant metrics including user satisfaction and commercial gains among others is that given a current baseline system, it is not obvious which aspect (such as diversity, relevance, etc.), nor which component (such as the ranker, candidate selector, serving infrastructure, etc.) should be the target of the next iteration. A common approach to address this is to develop well articulated hypotheses about issues (e.g. current serving approach has high latency leading to higher abandonment rate) or opportunities for improvement (e.g. point-wise ranker produces low diversity leading to very similar recommendations) in the current version, and use existing or new data to validate them. Unfortunately in many cases this is not possible, because the available data is not appropriate, or because gathering new data is too expensive."),(0,i.kt)("p",null,"Once a concrete improvement opportunity is detected, the current system is modified accordingly, for example by changing a prediction algorithm, the user interface or the serving infrastructure. These are usually local interventions with measurable local effects. But they also have (and they are expected to have) global or system-wide effects, which are a lot harder to measure since they involve the system as a whole. On top of this, there is usually tension between various variables of interest such as relevance vs latency, or diversity vs relevance or adaptivity vs user satisfaction to name a few which further increases the importance of system-wide analysis. The ability to make good and principled trade-offs is key to deliver a balanced and robust recommender system."))}m.isMDXComponent=!0}}]);