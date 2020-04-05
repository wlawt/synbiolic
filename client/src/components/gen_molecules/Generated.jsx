import React, { Component, Fragment } from "react";
import { Link } from "react-router-dom";
import { connect } from "react-redux";
import PropTypes from "prop-types";

import Chart from "react-apexcharts";
import { SlideDown } from "react-slidedown";
import "react-slidedown/lib/slidedown.css";

//import * as d3 from "d3";
//import FuzzySearch from 'fuzzy-search'

import Spinner from "../common/Spinner";

import {
  retrosynthesis_molecule,
  save_molecule_stats
} from "../../actions/moleculeActions";

var counter = -1;
class Generated extends Component {
  constructor(props) {
    super(props);

    this.state = {
      query: "",
      smiles: [],
      pic50: [],
      qed: [],
      display: false,
      success: false,
      error: false,
      filtered: [],

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

  onChange = e => {
    //e.preventDefault();

    let current = [];
    let new_list = [];

    if (e.target.value !== "" || this.state.search === "") {
      current = this.props.molecule.molecules.data.smiles;

      new_list = current.filter(item => {
        const lc = item.toLowerCase();

        const filter = e.target.value.toLowerCase();

        return lc.includes(filter);
      });
    } else {
      new_list = this.props.molecule.molecules.data.smiles;
    }

    this.setState({ filtered: new_list });

    //this.setState({ [e.target.name]: e.target.value });
  };

  onSearch = e => {
    e.preventDefault();

    /* var does_contain = this.state.smiles.includes(this.state.search);

    if (does_contain) {
      const { molecules } = this.props.molecule;
    } */
  };

  onLoadGraphData = () => {
    var series_data = [];
    /* for (var i = 0; i < this.props.molecule.molecules.data.pic50.length; i++) {
      series_data.push(this.props.molecule.molecules.data.pic50[i]);
    } */

    for (
      var i = 0;
      i < this.props.molecule.molecules.data.pic_dist.length;
      i++
    ) {
      series_data.push(this.props.molecule.molecules.data.pic_dist[i]);
    }

    const series_json = {
      name: "pic50 distribution",
      data: series_data
    };

    var cate_data = [];
    for (var j = 0; j < this.props.molecule.molecules.data.smiles.length; j++) {
      cate_data.push(j);
    }

    localStorage.removeItem("series");
    localStorage.removeItem("cate_data");
    localStorage.setItem("series", JSON.stringify(series_data));
    localStorage.setItem("cate_data", JSON.stringify(cate_data));

    // x-values
    this.setState(prevState => ({
      xaxis: {
        ...prevState.xaxis,
        ["categories"]: cate_data
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

    // Set states for graph
    if (this.state.series === null || this.state.series.length === 0) {
      this.onLoadGraphData();
    }
  };

  onRetro = (s, p, q) => {
    const data = {
      reactants:
        "C(O)1=C(O)C=C(C2=C(O)C=C([H])C([H])=C2C2=C([H])C=C([H])C=C2)C=C1O.[H].[H].[H]"
    };

    this.props.retrosynthesis_molecule(data);

    localStorage.removeItem("s_value");
    localStorage.removeItem("p_value");
    localStorage.removeItem("q_value");

    localStorage.setItem("s_value", s);
    localStorage.setItem("p_value", p);
    localStorage.setItem("q_value", q);

    this.props.history.push(`/retrosynthesis/${s}`);
  };

  onSave = (s, p, q) => {
    const data = {
      molecule: s,
      pic50: p,
      qed: q
    };

    try {
      this.props.save_molecule_stats(data);
      this.setState({ success: true });
    } catch (err) {
      console.log(err);
      this.setState({ error: true });
    }
  };

  onClose = e => {
    e.preventDefault();

    this.setState({ success: false });
  };

  onFail = e => {
    e.preventDefault();

    this.setState({ error: false });
  };

  render() {
    const { search } = this.state;
    const { molecules, loading } = this.props.molecule;
    let contents;

    if (molecules === null || loading) {
      contents = (
        <div className="h-100 row align-items-center">
          <div className="col text-center">
            <Spinner />
            <h3 className="title font-weight-bold mt-3">LOADING ...</h3>
          </div>
        </div>
      );
    } else {
      //console.log(molecules.data);
      if (this.state.filtered.length === 0) {
        this.setState({ filtered: molecules.data.smiles });
      }

      contents = (
        <Fragment>
          <h1 className="title font-weight-bold text-center">Generated Molecules</h1>

          <SlideDown className={"my-dropdown-slidedown"}>
            {this.state.success ? (
              <div class="row alert alert-success">
                <div class="col-md-6 text-center">
                  <span class="float-md-left">
                    Successfully saved molecule!
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

          <div className="form-group has-search pt-2">
            <form onSubmit={this.onSearch}>
              <span className="fa fa-search form-control-feedback"></span>
              <input
                type="text"
                className="form-control w-50"
                style={{ borderRadius: "25px" }}
                value={search}
                /* ref={input => this.search = input} */
                name="search"
                onChange={this.onChange}
                placeholder="Search"
              />
            </form>
          </div>

          <div className="row">
            <div className="col-md-6 text-center">
              <span className="float-md-left">
                <h5 className="subtitle font-weight-bold">
                  PIC50 Distribution Graph
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

          <div id="dataviz"></div>

          <div className="row mt-4">
            <div className="col-6 col-md-4">
              <button className="btn btn-primary btn-lg circled-btn w-100">
                Generated Molecules
              </button>
            </div>
            <div className="col-6 col-md-4">
              <button className="btn btn-primary btn-lg circled-btn w-100">
                Inhibition Activity (PIC50)
              </button>
            </div>
            <div className="col-6 col-md-4">
              <button className="btn btn-primary btn-lg circled-btn w-100">
                Drug Likeliness (QED)
              </button>
            </div>
          </div>
          <div className="table-responsive-sm mt-3">
            <table className="table table-boarded">
              <tbody className="subtitle font-weight-bold">
                {this.state.filtered.map(s => (
                  <tr key={counter++}>
                    <td>
                      <i
                        className="far fa-heart"
                        onClick={this.onSave.bind(
                          this,
                          s,
                          molecules.data.pic50[
                            molecules.data.smiles.indexOf(s)
                          ],
                          molecules.data.qed[molecules.data.smiles.indexOf(s)]
                        )}
                      ></i>{" "}
                      <Link
                        className="nocss"
                        to="#"
                        onClick={this.onRetro.bind(
                          this,
                          s,
                          molecules.data.pic50[
                            molecules.data.smiles.indexOf(s)
                          ],
                          molecules.data.qed[molecules.data.smiles.indexOf(s)]
                        )}
                      >
                        {s}
                      </Link>
                    </td>
                    <td className="text-center">
                      <p>
                        {molecules.data.pic50[molecules.data.smiles.indexOf(s)]}
                      </p>
                    </td>
                    <td className="text-center">
                      <p>
                        {molecules.data.qed[molecules.data.smiles.indexOf(s)]}
                      </p>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Fragment>
      );
    }

    return (
      <Fragment>
        <div className="container pt-2" style={{ marginBottom: "100px" }}>
          {contents}
        </div>
      </Fragment>
    );
  }
}

Generated.propTypes = {
  molecule: PropTypes.object.isRequired,
  errors: PropTypes.object.isRequired,
  retrosynthesis_molecule: PropTypes.func.isRequired,
  save_molecule_stats: PropTypes.func.isRequired
};

const mapStateToProps = state => ({
  molecule: state.molecule,
  errors: state.errors
});

export default connect(mapStateToProps, {
  retrosynthesis_molecule,
  save_molecule_stats
})(Generated);
