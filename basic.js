let dataset = [5, 10, 20, 30, 40, 50];
let height = 500;
let bar_width = 20;
let bar_padding = 10;
let width = dataset.length * (bar_width + bar_padding) - bar_padding;
let scale = d3.scaleLinear().domain([0, 50]).range([0, height]);
d3.select("svg").attr("width", width).attr("height", height);
let bar_chart = d3.select("svg").selectAll("rect")
    .data(dataset)
    .enter()
    .append("rect")
    .attr("y", function(d){return height - scale(d)})
    .attr("height", function(d){return scale(d)})
    .attr("width", bar_width)
    .attr("class", "bar")
    .attr("transform", function(d, i){return `translate(${(bar_width + bar_padding)* i}, 0)`});
