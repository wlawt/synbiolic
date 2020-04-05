import React, { Component, Fragment } from "react";
import { connect } from "react-redux";
import PropTypes from "prop-types";

import { get_molecule_stats } from "../../actions/moleculeActions";

import Spinner from "../common/Spinner";

import FuzzySearch from "fuzzy-search";

class saved extends Component {
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
    this.props.get_molecule_stats();
  }

  onSearch = e => {
    e.preventDefault();

    console.log("Searching");
  };

  onChange = e => {
    e.preventDefault();

    let current = [];
    let new_list = [];

    for (var i = 0; i < this.props.molecule.saved.data.length; i++) {
      current.push(`${this.props.molecule.saved.data[i].molecule}`);
    }

    if (e.target.value !== "" || this.state.search === "") {
      /* new_list = current.filter(item => {
        const lc = item.toLowerCase();

        const filter = e.target.value.toLowerCase();

        return lc.includes(filter);
      });

      var a = [];
      for (var i = 0; i < new_list.length; i++) {
        if (new_list[i] === this.props.molecule.saved.data[i].molecule) {
          var b = {
            molecule: new_list[i],
            pic50: this.props.molecule.saved.data[i].pic50,
            qed: this.props.molecule.saved.data[i].qed
          };
          a.push(b);
        }
      }

      console.log(a);
      new_list = a; */

      var result = [];
      const searcher = new FuzzySearch(
        this.props.molecule.saved.data,
        ["molecule"],
        {
          caseSensitive: false
        }
      );
      result = searcher.search(e.target.value);
      new_list = result;
    } else {
      new_list = this.props.molecule.saved.data;
    }

    this.setState({ filtered: new_list });
  };

  render() {
    const { search } = this.state;
    const { saved, loading } = this.props.molecule;
    let contents;

    if (saved === null || saved.length === 0 || loading) {
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
        this.setState({ filtered: saved.data });
      }

      //console.log(saved.data);
      contents = (
        <Fragment>
          <h1 className="title font-weight-bold text-center">Saved Molecules</h1>
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
                {this.state.filtered.map(data => (
                  <tr key={data._id}>
                    <td>{data.molecule}</td>
                    <td className="text-center">{data.pic50}</td>
                    <td className="text-center">{data.qed}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Fragment>
      );
    }

    //console.log(saved); -> data (_id, molecule, pic50, qed)
    return (
      <Fragment>
        <div className="container pt-2">{contents}</div>
      </Fragment>
    );
  }
}

saved.propTypes = {
  molecule: PropTypes.object.isRequired,
  get_molecule_stats: PropTypes.func.isRequired
};

const mapStateToProps = state => ({
  molecule: state.molecule
});

export default connect(mapStateToProps, { get_molecule_stats })(saved);
