import * as d3 from "d3"
import { useEffect, useState, useRef } from "react"
import { Box } from "@mantine/core"
import React from "react"
import _ from "lodash";

function tilegrid({ parameters, setAnswer }) {
  const ref = useRef(null)
  const { image, question, tiles, example, ourDefinition } = parameters
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [rectangles, setRectangles] = useState(tiles.map(t => ({ ...t, clicked: false, unClicked: false })))
  const containerRef = useRef(null);


  useEffect(() => {
    const img = new Image();
    img.src = image;
    img.onload = () => {
      const parent = containerRef.current;
      const availableHeight = window.innerHeight - parent.getBoundingClientRect().top; // Height from parent top to bottom of viewport
      const availableWidth = window.innerWidth - parent.getBoundingClientRect().left; // Width from parent left to right of viewport
      const aspectRatio = img.width / img.height; // Image aspect ratio

      let scaledWidth, scaledHeight;
      if (availableHeight * aspectRatio <= availableWidth) {
        // Fit height first
        scaledHeight = availableHeight;
        scaledWidth = availableHeight * aspectRatio;
      } else {
        // Fit width first
        scaledWidth = availableWidth;
        scaledHeight = availableWidth / aspectRatio;
      }
      setSize({ width: scaledWidth, height: scaledHeight, multiplier: scaledWidth / img.width });
    };
  }, [image]);


  useEffect(() => {
    const svg = d3.select(ref.current)
    const m = size.multiplier;
    svg
      .select("g")
      .selectAll("rect")
      .data(rectangles)
      .join("rect")
      .attr("x", d => d.x * m)
      .attr("y", d => d.y * m)
      .attr("width", d => d.width * m)
      .attr("height", d => d.height * m)
      .attr("stroke", "black")
      .attr("stroke-width", 2)
      .attr("cursor", "pointer")
      .attr("fill", "#e6e6e6")
      .attr("opacity", d => d.clicked ? 0 : 0.5)
      .on("click", (event, d) => {
        // change rectangles
        const newRectangles = _.cloneDeep(rectangles)
        const index = newRectangles.findIndex(item => item.x === d.x && item.y === d.y)
        if (newRectangles[index].clicked) newRectangles[index].unClicked = true
        newRectangles[index].clicked = !newRectangles[index].clicked
        setRectangles(newRectangles)

        setAnswer({
          status: true,
          answers: {
            patches: JSON.stringify({
              patches: newRectangles.filter(item => item.clicked||item.unClicked),
              multiplier: size.multiplier,
              question: question
            })
          }
        })
        
      })
      .on("mouseover", (event, d) => {
        d3.select(event.target).attr("stroke", "blue").attr("stroke-width", 2)
      })
      .on("mouseout", (event, d) => {
        d3.select(event.target).attr("stroke", "gray").attr("stroke-width", 0.5)
      });

  }, [rectangles, setAnswer, size])

  return (
    <div>
      {example && (
        <h1>Example Question</h1>
      )}

      {example ? (
        <div>
          <h3>{question}</h3>
        </div>
      ) : (
        <div>
          <h3>Please annotate <u>minimum patches</u> that <span style={{ color: "red" }} > must be revealed </span> for people to <span style={{ color: "red" }} >confidently</span> answer the question below, assuming that people <span style={{ color: "red" }} >cannot</span> see the blocked area. 
          <br/>Then provide your answer for the question.</h3>
          <h3>Q: {question}</h3>
        </div>
      )}

      <Box ref={containerRef} className="ImageWrapper" style={{ width: "100%", display: "block" }}>
        <svg id="clickAccuracySvg" ref={ref} width={size.width} height={size.height} >
          <image
            href={image}
            width={size.width}
            height={size.height}
          />
          <g id="rectangles"></g>
        </svg>
      </Box>
    </div>
  )
}

export default tilegrid
