import React, { Component, Fragment } from "react";
import { Link } from "react-router-dom";

import Home from "../img/home.png";
import Settings from "../img/settings.png";
import Flask from "../img/flask.png";
import Molecule from "../img/molecule.png";
/* import Book from "../img/book.png"; */
import Qna from "../img/qna.png";

class SideNav extends Component {
  render() {
    return (
      <Fragment>
        <nav className="col-md-1 d-none d-md-block sidebar">
          <div className="sidebar-sticky ">
            <ul className="nav flex-column mt-3">
              <li className="nav-item">
                <Link className="nav-link active" to="/main">
                  <img src={Home} width="50" height="50" alt="Home" />
                </Link>
              </li>
              <li className="nav-item">
                <Link className="nav-link" to="/generate">
                  <img src={Settings} width="50" height="50" alt="Settings" />
                </Link>
              </li>
              <li className="nav-item">
                <Link className="nav-link" to="/saved">
                  <img
                    src={Flask}
                    width="50"
                    height="50"
                    alt="Generate Molecules"
                  />
                </Link>
              </li>
              <li className="nav-item">
                <Link className="nav-link" to="/retro-request">
                  <img
                    src={Molecule}
                    width="50"
                    height="50"
                    alt="Generated Molecules"
                  />
                </Link>
              </li>
              {/*               <li className="nav-item">
                <Link className="nav-link" to="/generated">
                  <img
                    src={Book}
                    width="35"
                    height="35"
                    alt="List of Molecules"
                  />
                </Link>
              </li> */}
              <li className="nav-item">
                <Link className="nav-link" to="/qna">
                  <img
                    src={Qna}
                    width="35"
                    height="35"
                    alt="Support, Question & Answer"
                  />
                </Link>
              </li>
            </ul>
          </div>
        </nav>
      </Fragment>
    );
  }
}

export default SideNav;