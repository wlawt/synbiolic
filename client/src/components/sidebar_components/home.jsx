import React, { Component, Fragment } from "react";

class home extends Component {
  render() {
    return (
      <Fragment>
        <div className="container pt-2">
          <h1 className="text-center title font-weight-bold">
            Welcome to Synbiolic's Platform
          </h1>

          <div className="row mt-4">
            <div className="col-sm-6">
              <div className="card">
                <div className="card-header subtitle font-weight-bold">
                  Experiments
                </div>
                <div className="card-body">
                  <p className="card-text">
                    <b>Synbiolic is a online drug discovery platform.</b> which enables you to:</p>
                    <ul>
                      <li>
                      Generate novel drug molecules that demonstrate activity in target proteins.
                      </li>
                      <li>
                      Create retrosynthesis pathways for target molecules.
                      </li>
                      </ul> 
                      <p>Demo target is inhibiting the activities of JAK2, a mutation linked to leukemia.
                     
                  </p><a
                    target="_blank"
                    rel="noopener noreferrer"
                    href="/generate"
                    className="btn btn-primary btn-md circled-btn w-50"
                  >
                    Create Experiment
                  </a>
                </div>
                
              </div>
            </div>
            <div className="col-sm-6">
            <div className="card">
                <div className="card-header subtitle font-weight-bold">
                  Request New Target or Property
                </div>
                <div className="card-body">
                  <p className="card-text">
                    To reqest new target or property for your workspace, a
                    support plan is needed. If you are already on a support
                    plan, request below.
                  </p>
                  <a
                    target="_blank"
                    rel="noopener noreferrer"
                    href="https://docs.google.com/forms/d/e/1FAIpQLSccY7zzcniJvxRO-cAmdBjOSB9RXph3FEMchoZvqR2ERQl0LA/viewform"
                    className="btn btn-primary btn-md circled-btn w-50"
                  >
                    New Target/Property
                  </a>
                </div>
              </div>
              
            </div>
          </div>

          <div className="row mt-4">
            <div className="col-sm-6">
              <div className="card">
              <div className="card-header subtitle font-weight-bold">
                  Profile
                </div>
                <div className="card-body">
                  <p className="card-text">
                    <span className="font-weight-bold">Username:</span>{" "}
                    ImagineCupGuest2020
                  </p>
                  <p className="card-text">
                    <span className="font-weight-bold">Support Plan: </span> Basic
                  </p>
                  {/*<button className="btn btn-primary btn-md circled-btn w-25">
                    Login
                  </button> */}
                </div>
              </div>
            </div>
            <div className="col-sm-6">
            <div className="card">
              <div className="card-header subtitle font-weight-bold">
                  Recently Completed Experiment
                </div>
                <div className="card-body">
                  <p className="card-text">
                    <span className="font-weight-bold">Experiment Name:</span>{" "}
                    JAK2_pIC50
                  </p>
                  <p className="card-text">
                    <span className="font-weight-bold">Target:</span> JAK 2
                  </p>
                  <p className="card-text">
                    <span className="font-weight-bold">Property:</span>{" "}
                    Inhibition Activity
                  </p>
                  <p className="card-text">
                    <span className="font-weight-bold">
                      Molecules generated:
                    </span>{" "}
                    {localStorage.getItem("n_gen") === null
                      ? 0
                      : `${localStorage.getItem("n_gen")}`}
                  </p>
                </div>
                
              </div>
            </div>
          </div>
        </div>
      </Fragment>
    );
  }
}

export default home;
