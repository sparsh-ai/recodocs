/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docs: [
    'introduction',
    {
      type: 'category',
      label: 'Concept - Basics',
      link: {
        type: 'generated-index',
      },
    },
    {
      type: 'category',
      label: 'Concept - Extras',
      link: {
        type: 'generated-index',
      },
    },
  ],
};

module.exports = sidebars;
