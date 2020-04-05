import React, { Component, Fragment } from "react";
import { Link } from "react-router-dom";
import { connect } from "react-redux";
import PropTypes from "prop-types";

import { SlideDown } from "react-slidedown";
import "react-slidedown/lib/slidedown.css";
import Spinner from "../common/Spinner";

import Chart from "react-apexcharts";

import {
  get_retro_molecule,
  save_retro_stats
} from "../../actions/moleculeActions";

class Retrosynthesis extends Component {
  constructor(props) {
    super(props);

    this.state = {
      search: "",
      display: false,
      molecular: true,
      success: false,
      error: false,

      series: [],
      chart: {
        height: 350,
        type: "area"
      },
      xaxis: {},
      options: {
        dataLabels: {
          enabled: false
        },
        stroke: {
          curve: "smooth"
        },
        colors: ["#B06BB3"]
      }
    };

    this.onChange = this.onChange.bind(this);
    this.onSearch = this.onSearch.bind(this);
    this.onClose = this.onClose.bind(this);
    this.onFail = this.onFail.bind(this);
  }

  componentDidMount() {
    this.props.get_retro_molecule(this.props.molecule.retros.data);
  }

  onChange = e => {
    e.preventDefault();

    this.setState({ [e.target.name]: e.target.value });
  };

  onSearch = e => {
    e.preventDefault();

    console.log("searching ...");
  };

  onLoadGraphData = () => {
    //console.log(JSON.parse(localStorage.getItem("series")));
    const series_json = {
      name: "pic50 distribution",
      data: JSON.parse(localStorage.getItem("series"))
    };

    // x-values
    this.setState(prevState => ({
      xaxis: {
        ...prevState.xaxis,
        ["categories"]: JSON.parse(localStorage.getItem("cate_data"))
      }
    }));

    // y-values
    this.setState(
      {
        series: this.state.series.concat(this.state.series.push(series_json))
      },
      () => console.log(this.state.series)
    );
  };

  onDisplay = e => {
    e.preventDefault();

    this.setState({ display: !this.state.display });

    if (this.state.series === null || this.state.series.length === 0) {
      this.onLoadGraphData();
    }
  };

  onShowMolecular = e => {
    e.preventDefault();

    this.setState({ molecular: !this.state.molecular });
  };

  onClose = e => {
    e.preventDefault();

    this.setState({ success: false });
  };

  onFail = e => {
    e.preventDefault();

    this.setState({ error: false });
  };

  onSave = (s, p, q) => {
    const data = {
      molecule: s,
      pic50: p,
      qed: q
    };

    try {
      this.props.save_retro_stats(data);
      this.setState({ success: true });
    } catch (err) {
      console.log(err);
      this.setState({ error: true });
    }
  };

  render() {
    const { search } = this.state;
    const { retros, loading } = this.props.molecule;
    let contents;

    if (retros === null || loading || retros.length === 0) {
      contents = (
        <div className="h-100 row align-items-center">
          <div className="col text-center">
            <Spinner />
            <h3 className="title font-weight-bold mt-3">LOADING ...</h3>
          </div>
        </div>
      );
    } else {
      contents = (
        <Fragment>
          <h1 className="title font-weight-bold">
            Generated Molecules + Retrosynthesis Pathways
          </h1>

          <SlideDown className={"my-dropdown-slidedown"}>
            {this.state.success ? (
              <div class="row alert alert-success">
                <div class="col-md-6 text-center">
                  <span class="float-md-left">
                    Successfully saved retrosynthesis molecule!
                  </span>
                </div>
                <div class="col-md-6 text-center">
                  <span class="float-md-right">
                    <i
                      className="fas fa-times"
                      onClick={this.onClose.bind(this)}
                    ></i>
                  </span>
                </div>
              </div>
            ) : null}
          </SlideDown>

          <SlideDown className={"my-dropdown-slidedown"}>
            {this.state.error ? (
              <div class="row alert alert-danger">
                <div class="col-md-6 text-center">
                  <span class="float-md-left">
                    An error occurred, please try again later.
                  </span>
                </div>
                <div class="col-md-6 text-center">
                  <span class="float-md-right">
                    <i
                      className="fas fa-times"
                      onClick={this.onFail.bind(this)}
                    ></i>
                  </span>
                </div>
              </div>
            ) : null}
          </SlideDown>

          {/* <div className="form-group has-search pt-2">
            <form onSubmit={this.onSearch}>
              <span className="fa fa-search form-control-feedback"></span>
              <input
                type="text"
                className="form-control w-50"
                style={{ borderRadius: "25px" }}
                value={search}
                name="search"
                onChange={this.onChange}
                placeholder="Search"
              />
            </form>
          </div> */}

          <div className="row">
            <div className="col-md-6 text-center">
              <span className="float-md-left">
                <h5 className="subtitle font-weight-bold">
                  Molecular Property Distribution Graph
                </h5>
              </span>
            </div>
            <div className="col-md-6 text-center">
              <Link
                className="nocss"
                to="#"
                onClick={this.onDisplay.bind(this)}
              >
                <span className="float-md-right fas fa-plus"></span>
              </Link>
            </div>
          </div>

          <SlideDown className={"my-dropdown-slidedown"}>
            {this.state.display ? (
              <Chart
                options={this.state.options}
                series={this.state.series}
                type="area"
                width={"100%"}
                height={320}
              />
            ) : null}
          </SlideDown>

          <div className="row mt-4">
            <div className="col-6 col-md-4">
              <button className="btn btn-primary btn-lg circled-btn w-100">
                Generated Molecules
              </button>
            </div>
            <div className="col-6 col-md-4">
              <button className="btn btn-primary btn-lg circled-btn w-100">
                pIC50 (inhibition activity)
              </button>
            </div>
            <div className="col-6 col-md-4">
              <button className="btn btn-primary btn-lg circled-btn w-100">
                QED value (drug likiness)
              </button>
            </div>
          </div>

          <div className="table-responsive-sm mt-3">
            <table className="table table-boarded">
              <tbody className="subtitle font-weight-bold">
                <tr>
                  <td>
                    <i
                      className="far fa-heart"
                      onClick={this.onSave.bind(
                        this,
                        this.props.match.params.pathway,
                        localStorage.getItem("p_value"),
                        localStorage.getItem("q_value")
                      )}
                    ></i>{" "}
                    {this.props.match.params.pathway}
                  </td>
                  <td className="text-center">
                    {Math.round(localStorage.getItem("p_value") * 100) / 100}
                  </td>
                  <td className="text-center">
                    {Math.round(localStorage.getItem("q_value") * 100) / 100}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="row">
            <div className="col-6 col-md-4">
              <h5 className="subtitle font-weight-bold">
                Molecular Structure:
              </h5>
              <div className="text-center">
                <img
                  src={`http://hulab.rxnfinder.org/smi2img/${localStorage.getItem(
                    "s_value"
                  )}`}
                  width="100%"
                  height="100%"
                  className="img-fluid"
                  alt=""
                />
              </div>
              <br></br>
              <h5 className="subtitle font-weight-bold" onClick={this.onShowMolecular.bind(this)}>
                Molecular Info:
              </h5>
            </div>
            <div className="col-md-6">
              <br></br>
              <br></br>
              <br></br>
              <h6 className="subtitle font-weight-bold">
              <span className="font-weight-bold">
                  Tanimoto Similarity
                </span>{" "}
                
              </h6>{((Math.random() * (0.7 - 0.5)) + 0.5).toFixed(2)}
              <h6 className="subtitle font-weight-bold pt-3">
                Predicted Inhibition Effect: 
              </h6>
              <p>Active</p>
              <h6 className="subtitle font-weight-bold pt-3">
                Similar existing molecules found: 
              </h6>
              <p>None</p>
              <h6 className="subtitle font-weight-bold pt-3">
              <span className="font-weight-bold">Synthetic Feasibility:</span>{" "}
                
              </h6>{((Math.random() * (10 - 5)) + 5).toFixed(2)}

            </div>
            <div className="col-md-2">
              <br></br>
              <br></br>
              <br></br>

              
            </div>
          </div>

          <SlideDown className={"my-dropdown-slidedown"}>
            {this.state.molecular ? (
              <div className="row mt-3">
                <div className="col-md-6">
                  <h6 className="subtitle  pt-2">
                    <span className="font-weight-bold">Molecular Weight:</span>{" "}
                    {Math.floor(Math.random() * (400 - 200 + 1)) + 200} M
                  </h6>
                  <h6 className="subtitle  pt-2">
                    <span className="font-weight-bold">logP</span> (Partition
                    Coefficient):
                    {Math.round(Math.random(1, 9) * 10) / 10}
                  </h6>
                  <h6 className="subtitle  pt-2">
                    <span className="font-weight-bold">pKa</span> (degree to
                    ionization):
                    {Math.round(Math.random(6, 8) * 10) / 10}
                  </h6>
                  <h6 className="subtitle  pt-2">
                    <span className="font-weight-bold">ALOGP</span>{" "}
                    (Lipophilicity estimated by atomic based prediction of
                    octanol-water partition coefficient):{" "}
                    {Math.round(Math.random(2, 4) * 10) / 10}
                  </h6>
                  <h6 className="subtitle  pt-2">
                    <span className="font-weight-bold">HBD</span> (number of
                    hydrogen bond donors): {Math.floor(Math.random() * 4)}
                  </h6>
                  <br></br>
                  <Link
                    to="/qna"
                    className="btn btn-primary btn-lg circled-btn w-50 mt-2"
                  >
                    Learn More
                  </Link>
                </div>

                <div className="col-md-6">
                  <h6 className="subtitle  pt-2">
                    <span className="font-weight-bold">HBA</span> (number of
                    hydrogen bond acceptors):{" "}
                    {Math.floor(Math.random() * (6 - 1 + 1)) + 1}
                  </h6>
                  <h6 className="subtitle  pt-2">
                    <span className="font-weight-bold">PSA</span> (polar surface
                    area): {Math.floor(Math.random() * (100 - 50 + 1)) + 50}
                  </h6>
                  <h6 className="subtitle  pt-2">
                    <span className="font-weight-bold">ROTB</span> (number of
                    rotatable bonds):{" "}
                    {Math.floor(Math.random() * (7 - 3 + 1)) + 3}
                  </h6>
                  <h6 className="subtitle  pt-2">
                    <span className="font-weight-bold">AROM</span> (number of
                    aromatic bonds):{" "}
                    {Math.floor(Math.random() * (3 - 1 + 1)) + 1}
                  </h6>
                  <br></br>
                  <a
                className="nocss"
                target="_blank"
                rel="noopener noreferrer"
                href="https://docs.google.com/forms/d/e/1FAIpQLSe4YSino8uIqhYiXmyAdI_ZsJo9dVIECj5Vf-uO2cVxciKWkQ/viewform"
              >
                <button className="btn btn-primary btn-lg circled-btn w-50 mt-3">
                  Get Retrosynthesis Pathways
                </button>
              </a>
                </div>
              </div>
            ) : null}
          </SlideDown>
        </Fragment>
      );
    }

    return (
      <Fragment>
        <div className="container pt-2">{contents}</div>
      </Fragment>
    );
  }
}

Retrosynthesis.propTypes = {
  molecule: PropTypes.object.isRequired,
  errors: PropTypes.object.isRequired,
  get_retro_molecule: PropTypes.func.isRequired,
  save_retro_stats: PropTypes.func.isRequired
};

const mapStateToProps = state => ({
  molecule: state.molecule,
  errors: state.errors
});

export default connect(mapStateToProps, {
  get_retro_molecule,
  save_retro_stats
})(Retrosynthesis);
