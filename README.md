## Synbiolic Platform - Unlocking Drug Discovery with AI

Microsoft Imagine Cup Americas Regional Finalist & Runner-Up

## Acknowledgements

- [Deep reinforcement learning for de novo drug design Popova et. al 2018](https://advances.sciencemag.org/content/4/7/eaap7885)
- [DeepChem](https://github.com/deepchem/deepchem)
- [IBM RXN Chem](https://rxn.res.ibm.com/rxn/sign-in)
- [Azure ML](https://azure.microsoft.com/en-ca/services/machine-learning/)
- [Azure API Management](https://azure.microsoft.com/en-us/services/api-management/)
- [RXN Finder API](http://hulab.rxnfinder.org/smi2img/)

## Running the repo

### Downloading the source code

#### Clone the repository:

```
git clone https://github.com/wlawt/synbiolic.git
cd synbiolic-webapp
```

#### To run Synbiolic:

```
cd synbiolic
npm install
cd client
npm install
cd ../
npm run dev
```

## Instructions - Step-by-step Walkthrough:

#### Landing Page

![Landing page](https://github.com/wlawt/synbiolic/blob/master/client/src/components/img/landing.png)

#### User onboarding

![User onboarding](https://github.com/wlawt/synbiolic/blob/master/client/src/components/img/welcome.png)

#### Generate Molecules

![Generate Molecules](https://github.com/wlawt/synbiolic/blob/master/client/src/components/img/generate.png)

- The number of recommended molecules to generate is 20-50 (for demo purposes).
- It will then take you to all the generated molecules - where you can view the molecules and the plc-50 distribution
- You can save any of the molecules and click on them to bring up the retrosynthesis pathway
- The retrosynthesis pathway will have more details about the molecule itself

#### View saved molecules

![Saved Molecules](https://github.com/wlawt/synbiolic/blob/master/client/src/components/img/saved.png)

#### View Retrosynthesis Requests

![Retrosynthesis Requests](https://github.com/wlawt/synbiolic/blob/master/client/src/components/img/retro.png)

## Update Logs (user: wlawt)

#### April 5, 2020

- Making repo public

#### March 28, 2020

- Fixed pic50 distribution
- Added tab icon

#### March 27, 2020

- Fixed UI changes/bugs
- Corrected pic50 distribution chart
- Uninstalled unnecessary dep.

#### March 26, 2020

- UI Change, implemented 4th frame (sidebar, additional info for the retro, etc.)

#### March 25, 2020

- Added IBM Chemistry API for prediction/retrosynthesis

#### March 24, 2020

- Results show to front end
- Front end displaying data dynamically works
- Distribution chart for plc50 works
- Footer added
- Cleaned up code

#### March 21-23, 2020

- Implemented Azure API
- Fixed Azure CORS issue
- Showing results to front end

#### March 20, 2020

- Finished frame 3
- Fixed warnings

#### March 19, 2020

- Finished frame 1
- Finished frame 2
