import React, { Component, Fragment } from "react";
import PropTypes from "prop-types";
import { connect } from "react-redux";

import InputGroup from "../common/InputGroup";
import createImg from "../img/creating.svg";

import { Form } from "react-bootstrap";

import { generate_molecules } from "../../actions/moleculeActions";
class Generate extends Component {
  constructor(props) {
    super(props);

    this.state = {
      target: "",
      optimization: "",
      property: "",
      n_generate: ""
    };

    this.onChange = this.onChange.bind(this);
    this.onGenerate = this.onGenerate.bind(this);
  }

  onChange = e => {
    e.preventDefault();

    this.setState({ [e.target.name]: e.target.value });
  };

  onGenerate = e => {
    e.preventDefault();

    const json_res = {
      data: parseInt(this.state.n_generate)
    };

    localStorage.removeItem("n_gen");
    localStorage.setItem("n_gen", this.state.n_generate);

    this.props.generate_molecules(json_res);
    this.props.history.push("/generated");
  };

  render() {
    const { /* target, optimization, property, */ n_generate } = this.state;

    return (
      <Fragment>
        <div className="container">
          <div className="text-center">
            <h1 className="title font-weight-bold">Generate Molecules</h1>
          </div>

          <div className="row pt-5">
            <div className="col-md-6">
              <div className="px-2 text-left">
                <form
                  onSubmit={this.onGenerate}
                  className="justify-content-center"
                >
                 <Form.Label>
                    <b>Target Protein</b>
                    <p className="text-muted" style={{ marginBottom: "0" }}>
                      The protein that the generated molecules should bind
                      against.
                    </p>
                  </Form.Label>
                  <Form.Group className="w-75">
                    <Form.Control as="select" size="lg" custom>
                      <option>JAK 2</option>
                    </Form.Control>
                  </Form.Group>
                  <Form.Label>
                    <b>Optimization Type</b>
                    <p className="text-muted" style={{ marginBottom: "0" }}>
                      Maximize or minimize a specific property.
                    </p>
                  </Form.Label>
                  <Form.Group className="w-75">
                    <Form.Control as="select" size="lg" custom>
                      <option>PIC-50</option>
                    </Form.Control>
                  </Form.Group>
                  <Form.Label>
                    <b>Property to Optimize</b>
                    
                  </Form.Label>
                  <Form.Group className="w-75">
                    <Form.Control as="select" size="lg" custom>                      <option>Inhibition Activity</option>
                    </Form.Control>
                  </Form.Group>
                  <Form.Label>
                    <b># of Molecules to Generate</b>
                    <p className="text-muted" style={{ marginBottom: "0" }}>
                      Synbiolic recommends generating 50-500 molecules.
                    </p>
                  </Form.Label>
                  <InputGroup
                    label_desc="# of Molecules to Generate"
                    type="text"
                    value={n_generate}
                    name="n_generate"
                    onChange={this.onChange}
                  />

                  <button className="circled-btn w-75 btn-primary btn-lg mt-3">
                    Generate
                  </button>
                </form>
              </div>
            </div>
            <div className="smallnodisplay col-md-6 mt-4">
              <img
                src={createImg}
                width="90%"
                height="100%"
                className="img-fluid"
                alt=""
              />
            </div>
          </div>
        </div>
      </Fragment>
    );
  }
}

Generate.propTypes = {
  generate_molecules: PropTypes.func.isRequired,
  molecule: PropTypes.object.isRequired
};

const mapStateToProps = state => ({
  errors: state.errors,
  molecule: state.molecule
});

export default connect(mapStateToProps, { generate_molecules })(Generate);
