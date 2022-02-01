
import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug','3d6'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config','914'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content','c28'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData','3cf'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata','31b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry','0da'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes','244'),
    exact: true
  },
  {
    path: '/blog',
    component: ComponentCreator('/blog','8ad'),
    exact: true
  },
  {
    path: '/blog/archive',
    component: ComponentCreator('/blog/archive','f4c'),
    exact: true
  },
  {
    path: '/blog/first-blog-post',
    component: ComponentCreator('/blog/first-blog-post','b1a'),
    exact: true
  },
  {
    path: '/blog/long-blog-post',
    component: ComponentCreator('/blog/long-blog-post','6d5'),
    exact: true
  },
  {
    path: '/blog/mdx-blog-post',
    component: ComponentCreator('/blog/mdx-blog-post','bee'),
    exact: true
  },
  {
    path: '/blog/tags',
    component: ComponentCreator('/blog/tags','e13'),
    exact: true
  },
  {
    path: '/blog/tags/docusaurus',
    component: ComponentCreator('/blog/tags/docusaurus','f5a'),
    exact: true
  },
  {
    path: '/blog/tags/facebook',
    component: ComponentCreator('/blog/tags/facebook','134'),
    exact: true
  },
  {
    path: '/blog/tags/hello',
    component: ComponentCreator('/blog/tags/hello','6d0'),
    exact: true
  },
  {
    path: '/blog/tags/hola',
    component: ComponentCreator('/blog/tags/hola','18c'),
    exact: true
  },
  {
    path: '/blog/welcome',
    component: ComponentCreator('/blog/welcome','351'),
    exact: true
  },
  {
    path: '/markdown-page',
    component: ComponentCreator('/markdown-page','be1'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs','128'),
    routes: [
      {
        path: '/docs/concept-basics/types-of-recommender-systems',
        component: ComponentCreator('/docs/concept-basics/types-of-recommender-systems','5b7'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/1mg-prod2vec',
        component: ComponentCreator('/docs/concept-extras/case-studies/1mg-prod2vec','9e2'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/airbnb-experiences',
        component: ComponentCreator('/docs/concept-extras/case-studies/airbnb-experiences','4cd'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/alipay-ctr',
        component: ComponentCreator('/docs/concept-extras/case-studies/alipay-ctr','850'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/doordash-contextual-bandit',
        component: ComponentCreator('/docs/concept-extras/case-studies/doordash-contextual-bandit','32a'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/etsy-personalization',
        component: ComponentCreator('/docs/concept-extras/case-studies/etsy-personalization','d90'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/huawei-appgallery',
        component: ComponentCreator('/docs/concept-extras/case-studies/huawei-appgallery','9bd'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/linkedin-glmix',
        component: ComponentCreator('/docs/concept-extras/case-studies/linkedin-glmix','5ae'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/marketcloud-real-time',
        component: ComponentCreator('/docs/concept-extras/case-studies/marketcloud-real-time','d26'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/netflix-personalize-images',
        component: ComponentCreator('/docs/concept-extras/case-studies/netflix-personalize-images','62a'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/pinterest-multi-task-learning',
        component: ComponentCreator('/docs/concept-extras/case-studies/pinterest-multi-task-learning','9f7'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/santander-banking-products',
        component: ComponentCreator('/docs/concept-extras/case-studies/santander-banking-products','05a'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/scribd-real-time',
        component: ComponentCreator('/docs/concept-extras/case-studies/scribd-real-time','8e4'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/spotify-contextual-bandits',
        component: ComponentCreator('/docs/concept-extras/case-studies/spotify-contextual-bandits','54c'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/spotify-rl',
        component: ComponentCreator('/docs/concept-extras/case-studies/spotify-rl','2f5'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/stitchfix-multi-armed-bandit',
        component: ComponentCreator('/docs/concept-extras/case-studies/stitchfix-multi-armed-bandit','24c'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/taobao-bst',
        component: ComponentCreator('/docs/concept-extras/case-studies/taobao-bst','3dd'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/ubereats-personalization',
        component: ComponentCreator('/docs/concept-extras/case-studies/ubereats-personalization','d7a'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/case-studies/walmart-model-selection',
        component: ComponentCreator('/docs/concept-extras/case-studies/walmart-model-selection','b7a'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/graph-embeddings',
        component: ComponentCreator('/docs/concept-extras/graph-embeddings','91f'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/concept-extras/incremental-learning',
        component: ComponentCreator('/docs/concept-extras/incremental-learning','a40'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/datasets',
        component: ComponentCreator('/docs/datasets','9a1'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/intro',
        component: ComponentCreator('/docs/intro','99a'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/',
        component: ComponentCreator('/docs/models/','0c1'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/a3c',
        component: ComponentCreator('/docs/models/a3c','031'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/afm',
        component: ComponentCreator('/docs/models/afm','45f'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/afn',
        component: ComponentCreator('/docs/models/afn','f97'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/ar',
        component: ComponentCreator('/docs/models/ar','033'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/asmg',
        component: ComponentCreator('/docs/models/asmg','7e3'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/attrec',
        component: ComponentCreator('/docs/models/attrec','f97'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/autoint',
        component: ComponentCreator('/docs/models/autoint','6ef'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/bcq',
        component: ComponentCreator('/docs/models/bcq','350'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/biasonly',
        component: ComponentCreator('/docs/models/biasonly','227'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/bpr',
        component: ComponentCreator('/docs/models/bpr','204'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/caser',
        component: ComponentCreator('/docs/models/caser','ab3'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/dcn',
        component: ComponentCreator('/docs/models/dcn','bc7'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/ddpg',
        component: ComponentCreator('/docs/models/ddpg','5f5'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/deepcross',
        component: ComponentCreator('/docs/models/deepcross','3b9'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/deepfm',
        component: ComponentCreator('/docs/models/deepfm','aca'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/deepwalk',
        component: ComponentCreator('/docs/models/deepwalk','65b'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/dgtn',
        component: ComponentCreator('/docs/models/dgtn','7f1'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/dqn',
        component: ComponentCreator('/docs/models/dqn','5a9'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/drqn',
        component: ComponentCreator('/docs/models/drqn','e87'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/drr',
        component: ComponentCreator('/docs/models/drr','d43'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/dueling-dqn',
        component: ComponentCreator('/docs/models/dueling-dqn','ecd'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/ffm',
        component: ComponentCreator('/docs/models/ffm','5b3'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/fgnn',
        component: ComponentCreator('/docs/models/fgnn','f7d'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/fm',
        component: ComponentCreator('/docs/models/fm','1cf'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/gat',
        component: ComponentCreator('/docs/models/gat','41f'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/gc-san',
        component: ComponentCreator('/docs/models/gc-san','9e6'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/gce-gnn',
        component: ComponentCreator('/docs/models/gce-gnn','c76'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/gru4rec',
        component: ComponentCreator('/docs/models/gru4rec','fe2'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/hmlet',
        component: ComponentCreator('/docs/models/hmlet','35c'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/itempop',
        component: ComponentCreator('/docs/models/itempop','e0a'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/lessr',
        component: ComponentCreator('/docs/models/lessr','1a3'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/lightfm-warp',
        component: ComponentCreator('/docs/models/lightfm-warp','810'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/lightgcn',
        component: ComponentCreator('/docs/models/lightgcn','9d0'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/lird',
        component: ComponentCreator('/docs/models/lird','556'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/markov-chains',
        component: ComponentCreator('/docs/models/markov-chains','645'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/mb-gmn',
        component: ComponentCreator('/docs/models/mb-gmn','6c8'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/mf',
        component: ComponentCreator('/docs/models/mf','7cd'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/mian',
        component: ComponentCreator('/docs/models/mian','77e'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/neumf',
        component: ComponentCreator('/docs/models/neumf','b00'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/nfm',
        component: ComponentCreator('/docs/models/nfm','338'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/ngcf',
        component: ComponentCreator('/docs/models/ngcf','c19'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/pnn',
        component: ComponentCreator('/docs/models/pnn','87c'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/ppo',
        component: ComponentCreator('/docs/models/ppo','043'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/q-learning',
        component: ComponentCreator('/docs/models/q-learning','00c'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/sac',
        component: ComponentCreator('/docs/models/sac','959'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/sarsa',
        component: ComponentCreator('/docs/models/sarsa','00a'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/sasrec',
        component: ComponentCreator('/docs/models/sasrec','613'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/sgl',
        component: ComponentCreator('/docs/models/sgl','90e'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/siren',
        component: ComponentCreator('/docs/models/siren','08f'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/slist',
        component: ComponentCreator('/docs/models/slist','8c2'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/spop',
        component: ComponentCreator('/docs/models/spop','5ea'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/sr',
        component: ComponentCreator('/docs/models/sr','47f'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/sr-gnn',
        component: ComponentCreator('/docs/models/sr-gnn','2cf'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/sr-san',
        component: ComponentCreator('/docs/models/sr-san','3cb'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/stamp',
        component: ComponentCreator('/docs/models/stamp','13f'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/svae',
        component: ComponentCreator('/docs/models/svae','c85'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/tagnn',
        component: ComponentCreator('/docs/models/tagnn','f52'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/tagnn-pp',
        component: ComponentCreator('/docs/models/tagnn-pp','e79'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/vncf',
        component: ComponentCreator('/docs/models/vncf','381'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/vsknn',
        component: ComponentCreator('/docs/models/vsknn','cd4'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/wide-and-deep',
        component: ComponentCreator('/docs/models/wide-and-deep','de0'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/word2vec',
        component: ComponentCreator('/docs/models/word2vec','908'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/models/xdeepfm',
        component: ComponentCreator('/docs/models/xdeepfm','8bf'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/projects',
        component: ComponentCreator('/docs/projects','c4a'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/tutorials/addressing-the-cold-start-problem',
        component: ComponentCreator('/docs/tutorials/addressing-the-cold-start-problem','675'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/tutorials/data-science-bookcamp',
        component: ComponentCreator('/docs/tutorials/data-science-bookcamp','f86'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/tutorials/matching-and-ranking-models-in-tensorflow',
        component: ComponentCreator('/docs/tutorials/matching-and-ranking-models-in-tensorflow','646'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/tutorials/real-time-event-capturing-with-kafka-and-mongodb',
        component: ComponentCreator('/docs/tutorials/real-time-event-capturing-with-kafka-and-mongodb','852'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/tutorials/session-based-recommendation-with-graph-neural-net',
        component: ComponentCreator('/docs/tutorials/session-based-recommendation-with-graph-neural-net','fda'),
        exact: true,
        'sidebar': "tutorialSidebar"
      },
      {
        path: '/docs/tutorials/tensorflow-2-reinforcement-learning-cookbook',
        component: ComponentCreator('/docs/tutorials/tensorflow-2-reinforcement-learning-cookbook','646'),
        exact: true,
        'sidebar': "tutorialSidebar"
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/','deb'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*')
  }
];
