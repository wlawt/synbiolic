import React, { Component } from "react";
import { BrowserRouter as Router, Route, Redirect } from "react-router-dom";

import { Provider } from "react-redux";
import store from "./store";

import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

// IMPORT COMPONENTS
import Navbar from "./components/layout/Navbar";
import SideNav from "./components/layout/SideNav";
import Generate from "./components/gen_molecules/Generate";
import Generated from "./components/gen_molecules/Generated";
import Retrosynthesis from "./components/gen_molecules/Retrosynthesis";
import Home from "./components/sidebar_components/home";
import Qna from "./components/sidebar_components/qna";
import Saved from "./components/sidebar_components/saved";
import Request from "./components/sidebar_components/request";
import Footer from "./components/layout/Footer";

class App extends Component {
  render() {
    return (
      <Provider store={store}>
        <Router>
          <div className="App">
            <Navbar />
            <div className="row sidenav">
              <SideNav />

              <div className="main mt-4 col-lg-10">
                <Route exact path="/generate" component={Generate} />
                <Route exact path="/generated" component={Generated} />
                <Route exact path="/qna" component={Qna} />
                <Route exact path="/saved" component={Saved} />
                <Route exact path="/retro-request" component={Request} />
                <Route exact path="/main" component={Home} />

                <Route
                  exact
                  path="/retrosynthesis/:pathway"
                  component={Retrosynthesis}
                />

                <Route
                  exact
                  path="/"
                  /* render={() => <Redirect to="/home.html" />} */
                  component={() => {
                    window.location.href = "/home.html";
                    return null;
                  }}
                />
              </div>
            </div>
            <Footer />
          </div>
        </Router>
      </Provider>
    );
  }
}

export default App;