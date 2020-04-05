import React, { Component, Fragment } from "react";

import { SlideDown } from "react-slidedown";
import "react-slidedown/lib/slidedown.css";

class qna extends Component {
  constructor() {
    super();

    this.state = {
      display1: true,
      display2: true,
      display3: true
    };

    this.onChange = this.onChange.bind(this);
  }

  onChange = e => {
    this.setState({ [e.target.name]: e.target.value });
  };

  onQ1 = e => {
    e.preventDefault();

    this.setState({ display1: !this.state.display1 });
  };
  onQ2 = e => {
    e.preventDefault();

    this.setState({ display2: !this.state.display2 });
  };
  onQ3 = e => {
    e.preventDefault();

    this.setState({ display3: !this.state.display3 });
  };

  render() {
    return (
      <Fragment>
        <div className="container pt-2" style={{ marginBottom: "200px" }}>
          <h1 className="title font-weight-bold text-center">
            Commonly Asked Questions
          </h1>

          <div className="card mt-5">
            <div
              className="card-header subtitle font-weight-bold"
              onClick={this.onQ1.bind(this)}
            >
              How do you generate small molecule?
            </div>
            <SlideDown className={"my-dropdown-slidedown"}>
              {this.state.display1 ? (
                <ul className="list-group list-group-flush">
                  <li className="list-group-item subtitle">
                    If you haven’t yet set up your workspace and selected the
                    target protein and property associated to your workspace,
                    feel free to contact us below to do so. To start an
                    experiment and generate small molecules, click on the
                    gear/experiment icon on the left to get started.
                  </li>
                </ul>
              ) : null}
            </SlideDown>
          </div>

          <div className="card mt-1">
            <div
              className="card-header subtitle font-weight-bold"
              onClick={this.onQ2.bind(this)}
            >
              What does the QED value represent?
            </div>
            <SlideDown className={"my-dropdown-slidedown"}>
              {this.state.display2 ? (
                <ul className="list-group list-group-flush">
                  <li className="list-group-item subtitle">
                    The QED value represents the drug likeliness of a small
                    molecule and is a key consideration when selecting potential
                    drug compounds. A small molecule is considered as good
                    quality should have a QED value >0.5. Molecules generated on
                    our platform can achieve a QED value >0.5 majority of the
                    time.
                  </li>
                </ul>
              ) : null}
            </SlideDown>
          </div>

          <div className="card mt-1">
            <div
              className="card-header subtitle font-weight-bold"
              onClick={this.onQ3.bind(this)}
            >
              How do I request instructions on how to synhtesize generated small
              molecules?
            </div>
            <SlideDown className={"my-dropdown-slidedown"}>
              {this.state.display3 ? (
                <ul className="list-group list-group-flush">
                  <li className="list-group-item subtitle">
                    To do so, please send a reqest to our retrosynthesis
                    platform. There is a button to do so when you click on one
                    of the generated molecules.
                  </li>
                </ul>
              ) : null}
            </SlideDown>
          </div>

          <div className="row mt-5">
            <div className="col-sm-12">
              <div className="text-center">
                <a
                  target="_blank"
                  rel="noopener noreferrer"
                  href="https://docs.google.com/forms/d/e/1FAIpQLSccY7zzcniJvxRO-cAmdBjOSB9RXph3FEMchoZvqR2ERQl0LA/viewform"
                  className="btn btn-primary btn-md circled-btn w-25"
                >
                  Contact Us
                </a>
              </div>
            </div>
          </div>
        </div>
      </Fragment>
    );
  }
}

export default qna;
