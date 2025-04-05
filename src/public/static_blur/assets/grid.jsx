import * as d3 from "d3"
import { useEffect, useState, useRef } from "react"
import { Box } from "@mantine/core"
import React from "react"
import _ from "lodash";

function Grid({ parameters, setAnswer }) {
  const ref = useRef(null)
  const { image, question, x_grids, y_grids, example } = parameters
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [rectangles, setRectangles] = useState([])
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

      let pad_x_grids = x_grids;
      let pad_y_grids = y_grids;
      if (!x_grids.includes(0)) pad_x_grids = [0, ...x_grids]
      if (!y_grids.includes(0)) pad_y_grids = [0, ...y_grids]
      const rect = []
      for (let i = 0; i < pad_x_grids.length; i++) {
        for (let j = 0; j < pad_y_grids.length; j++) {
          let width, height;
          if (i + 1 === pad_x_grids.length) width = img.width - pad_x_grids[i]
          else width = pad_x_grids[i + 1] - pad_x_grids[i]
          if (j + 1 === pad_y_grids.length) height = img.height - pad_y_grids[j]
          else height = pad_y_grids[j + 1] - pad_y_grids[j]

          rect.push({
            x: pad_x_grids[i], // Top-left x-coordinate
            y: pad_y_grids[j], // Top-left y-coordinate
            width: width, // Width of the rectangle
            height: height, // Height of the rectangle,
            clicked: false,
            unClicked: false
          })

        }
      }
      setRectangles(rect)
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
          Then provide your answer for the question.</h3>
          <h3>Q: {question}</h3>
        </div>
      )}

      <Box ref={containerRef} className="ImageWrapper" style={{ width: "100%", display: "block" }}>
        <svg id="clickAccuracySvg" ref={ref} width={size.width} height={size.height} >
          <defs>
            <filter id="imageBlurFilter" >
              <feGaussianBlur in="SourceGraphic" stdDeviation="17" />
              <feComponentTransfer>
                <feFuncA type="table" tableValues="1 1" />
              </feComponentTransfer>
            </filter>

            <mask id="unblurMask">
              <rect width="100%" height="100%" fill="white" />
              {rectangles.filter(r => r.clicked).map((d, i) => (
                <rect
                  key={i}
                  x={d.x * size.multiplier}
                  y={d.y * size.multiplier}
                  width={d.width * size.multiplier}
                  height={d.height * size.multiplier}
                  fill="black"
                />
              ))}
            </mask>
          </defs>
          <image
            href={image}
            width={size.width}
            height={size.height}
          />
          {/* <image
            href={image}
            width={size.width}
            height={size.height}
            filter="url(#imageBlurFilter)"
            mask="url(#unblurMask)"
          /> */}
          <g id="rectangles"></g>
        </svg>
      </Box>
    </div>
  )
}

export default Grid
