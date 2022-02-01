"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[7978],{3905:function(e,t,n){n.d(t,{Zo:function(){return u},kt:function(){return f}});var r=n(7294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var c=r.createContext({}),d=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},u=function(e){var t=d(e.components);return r.createElement(c.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},l=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,i=e.originalType,c=e.parentName,u=s(e,["components","mdxType","originalType","parentName"]),l=d(n),f=o,m=l["".concat(c,".").concat(f)]||l[f]||p[f]||i;return n?r.createElement(m,a(a({ref:t},u),{},{components:n})):r.createElement(m,a({ref:t},u))}));function f(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var i=n.length,a=new Array(i);a[0]=l;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s.mdxType="string"==typeof e?e:o,a[1]=s;for(var d=2;d<i;d++)a[d]=n[d];return r.createElement.apply(null,a)}return r.createElement.apply(null,n)}l.displayName="MDXCreateElement"},7987:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return s},contentTitle:function(){return c},metadata:function(){return d},toc:function(){return u},default:function(){return l}});var r=n(7462),o=n(3366),i=(n(7294),n(3905)),a=["components"],s={},c="Spotify Contextual Bandits",d={unversionedId:"concept-extras/case-studies/spotify-contextual-bandits",id:"concept-extras/case-studies/spotify-contextual-bandits",title:"Spotify Contextual Bandits",description:"Read more in this paper",source:"@site/docs/concept-extras/case-studies/spotify-contextual-bandits.md",sourceDirName:"concept-extras/case-studies",slug:"/concept-extras/case-studies/spotify-contextual-bandits",permalink:"/docs/concept-extras/case-studies/spotify-contextual-bandits",editUrl:"https://github.com/recohut/docs/docs/docs/concept-extras/case-studies/spotify-contextual-bandits.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Scribd Real-time",permalink:"/docs/concept-extras/case-studies/scribd-real-time"},next:{title:"Spotify RL",permalink:"/docs/concept-extras/case-studies/spotify-rl"}},u=[],p={toc:u};function l(e){var t=e.components,n=(0,o.Z)(e,a);return(0,i.kt)("wrapper",(0,r.Z)({},p,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"spotify-contextual-bandits"},"Spotify Contextual Bandits"),(0,i.kt)("p",null,(0,i.kt)("a",{parentName:"p",href:"https://dl.acm.org/doi/10.1145/3240323.3240354"},"Read more in this paper")),(0,i.kt)("p",null,"Spotify using contextual bandits to identify the best recommendation explanation (aka \u201crecsplanations\u201d) for users. The problem was how to jointly personalize music recommendations with their associated explanation, where the reward is user engagement on the recommendation. Contextual features include user region, product, and platform of the user device, user listening history (genres, playlist), etc."),(0,i.kt)("p",null,"An initial approach involved using logistic regression to predict user engagement from a recsplanation, given data about the recommendation, explanation, and user context. However, for logistic regression, the recsplanation that maximized reward was the same regardless of the user context."),(0,i.kt)("p",null,"To address this, they introduced higher-order interactions between recommendation, explanation, and user context, first by embedding them, and then introducing inner products on the embeddings (i.e., 2nd-order interactions). Then, the 2nd-order interactions are combined with first-order variables via a weighted sum, making it a 2nd-order factorization machine. They tried both 2nd and 3rd order factorization machines. (For more details of factorization machines in recommendations, see figure 2 and the \u201cFM Component\u201d section in the ",(0,i.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/1703.04247"},"DeepFM")," paper)."),(0,i.kt)("p",null,"To train their model, they adopt sample reweighting to account for the non-uniform probability of recommendations in production. (They didn\u2019t have the benefit of uniform random samples like in the Netflix example.) During the offline evaluation, the 3rd-order factorization machine performed best. During online evaluation (i.e., A/B test), both 2nd and 3rd-order factorization machines did better than logistic regression and the baseline. Nonetheless, there was no significant difference between the 2nd and 3rd-order models."))}l.isMDXComponent=!0}}]);