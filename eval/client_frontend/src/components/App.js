import './App.css';
import React from 'react';
import axios from 'axios';
import update from 'immutability-helper';

var mouseDown = 0;
document.body.onmousedown = function() { 
  ++mouseDown;
}
document.body.onmouseup = function() {
  mouseDown=0;
}

// function App() {
class App extends React.Component {
  constructor(props){
    super(props);
    this.state = {image: [ /* 28x28 */
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
      [0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0],
    ]};
    this.handleClick = this.handleClick.bind(this);
    this.classify = this.classify.bind(this);
    this.setStyle = this.setStyle.bind(this);
  }

  setStyle(x,y) {
    let backgroundColor_ = 'white'
    if (this.state.image[x][y] > 0){
      backgroundColor_ = 'black'
    }
    return {border: '0px solid black', backgroundColor: backgroundColor_, width: '10px', height: '10px'}
  }

  handleClick(x,y) {
    if (mouseDown > 0){
      let newArray = update(this.state.image, {
        [x]: {
          [y]: {$set: 1}
        }
      })
      this.setState({image: newArray})
    }
  }

  classify() {
    console.log('classify')
    axios.post('http://localhost:9999/', {image: this.state.image})
        .then(response => {
          console.log("response\t" + response.data)
    });
  }

  render() {
    return (
      <table>
        <tr>
          <td style={{paddingLeft: '50px'}}>
            <div className="App" draggable="false" style={{marginTop: '50px'}}>
              <table style={{border: '1px solid black', borderSpacing: 0}}>
                {[...Array(25)].map((x, i) =>
                  <tr style={{border: '0px solid black'}}>
                  {[...Array(25)].map((x, y) =>
                    <td style={this.setStyle(i,y)} onMouseOver={() => this.handleClick(i,y)}></td>
                  )}
                  </tr>
                )}
              </table>
            </div>
          </td>
          <td style={{paddingLeft: '50px'}}>
            <button style={{padding: '10px 30px 10px 30px'}} onClick={this.classify}>Classify</button>
          </td>
        </tr>
      </table>
    );
  }
}

export default App;
