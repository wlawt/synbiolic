import React, { Component, Fragment } from "react";
import { connect } from "react-redux";
import PropTypes from "prop-types";
import { Link } from "react-router-dom";

import { get_retro_stats } from "../../actions/moleculeActions";

import Spinner from "../common/Spinner";

import FuzzySearch from "fuzzy-search";

class request extends Component {
  constructor(props) {
    super(props);

    this.state = {
      query: "",
      filtered: []
    };

    this.onSearch = this.onSearch.bind(this);
    this.onChange = this.onChange.bind(this);
  }

  componentDidMount() {
    this.props.get_retro_stats();
  }

  onSearch = e => {
    e.preventDefault();

    console.log("Searching");
  };

  onChange = e => {
    e.preventDefault();

    let current = [];
    let new_list = [];

    for (var i = 0; i < this.props.molecule.retro_stats.data.length; i++) {
      current.push(`${this.props.molecule.retro_stats.data[i].molecule}`);
    }

    if (e.target.value !== "" || this.state.search === "") {
      var result = [];
      const searcher = new FuzzySearch(
        this.props.molecule.retro_stats.data,
        ["molecule"],
        {
          caseSensitive: false
        }
      );
      result = searcher.search(e.target.value);
      new_list = result;
    } else {
      new_list = this.props.molecule.retro_stats.data;
    }

    this.setState({ filtered: new_list });
  };

  render() {
    const { search } = this.state;
    const { retro_stats, loading } = this.props.molecule;
    let contents;

    if (retro_stats === null || retro_stats.length === 0 || loading) {
      contents = (
        <div className="h-100 row align-items-center">
          <div className="col text-center">
            <Spinner />
            <h3 className="title font-weight-bold mt-3">LOADING ...</h3>
          </div>
        </div>
      );
    } else {
      if (this.state.filtered.length === 0) {
        this.setState({ filtered: retro_stats.data });
      }

      contents = (
        <Fragment>
          <h1 className="title font-weight-bold text-center">Retrosynthesis Request</h1>
          <div className="form-group has-search pt-2">
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
          </div>

          <div className="row">
            <h1 className="title">Target: JAK2</h1>
          </div>

          <div className="row mt-4 row-cols-1 row-cols-sm-2 row-cols-md-4">
            <div className="col">
              <button className="btn btn-primary btn-lg circled-btn w-100">
                Generated Molecules
              </button>
            </div>
            <div className="col">
              <button className="btn btn-primary btn-lg circled-btn w-100">
              Inhibition Activity
              </button>
            </div>
            <div className="col">
              <button className="btn btn-primary btn-lg circled-btn w-100">
              Drug Likeliness
              </button>
            </div>
            <div className="col">
              <button className="btn btn-primary btn-lg circled-btn w-100">
                Status
              </button>
            </div>
          </div>

          <div className="table-responsive-sm mt-3">
            <table className="table table-boarded">
              <tbody className="subtitle font-weight-bold">
                {retro_stats.data.length !== 0 ? (
                  <Fragment>
                    {this.state.filtered.map(data => (
                      <tr
                        key={data._id}
                        className="row row-cols-1 row-cols-sm-2 row-cols-md-4"
                      >
                        <td className="col text-center">{data.molecule}</td>
                        <td className="col text-center">{data.pic50}</td>
                        <td className="col text-center">{data.qed}</td>
                        <td className="col text-center">In Progress</td>
                      </tr>
                    ))}
                  </Fragment>
                ) : (
                  <h4 className="subtitle text-center mt-3">
                    No retrosynthesis paths saved.
                    <Link to="/generate" className="nocss">
                      Create some molecules.
                    </Link>
                  </h4>
                )}
              </tbody>
            </table>
          </div>
        </Fragment>
      );
    }

    return (
      <Fragment>
        <div className="container pt-2" style={{ marginBottom: "200px" }}>
          {contents}
        </div>
      </Fragment>
    );
  }
}

request.propTypes = {
  molecule: PropTypes.object.isRequired,
  get_retro_stats: PropTypes.func.isRequired
};

const mapStateToProps = state => ({
  molecule: state.molecule
});

export default connect(mapStateToProps, { get_retro_stats })(request);
