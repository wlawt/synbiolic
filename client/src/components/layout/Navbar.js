import React, { Component } from "react";
/* import { Link } from "react-router-dom";
import PropTypes from "prop-types";
import { connect } from "react-redux"; */

import Syn from "../img/syn.png";

class Navbar extends Component {
  render() {
    return (
      <nav className="navbg navbar navbar-expand-sm navbar-dark bg-dark">
        <div className="container">
          <div className="row">
            <a className="navbar-brand mt-2 mb-2" href="/home.html">
              <img src={Syn} width="175" height="75" alt="" />
            </a>
          </div>
        </div>
      </nav>
    );
  }
}

export default Navbar;
