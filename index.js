let model;
let contentFileSelect = document.getElementById('content-file-selector');
let styleFileSelect = document.getElementById('style-file-selector');
let content = document.getElementById('content-image');
let style = document.getElementById('style-image');
let alpha = tf.tensor2d([[1.0]]);
let strengthSlider = document.getElementById('stylization-intensity');
let stylizedCanvas = document.getElementById('stylized-image');
let StatusElement = document.getElementById("time");
const time = msg => StatusElement.innerText = msg;



/* Load model function */
async function loadModel(name) 
{
    $(".progress-bar").show();
    model = undefined;
    model = await tf.loadGraphModel('web_model/model.json');
    $('.progress-bar').hide();
}


/* VGG19 preprocess */
async function preprocess(inputs)
{
    const VGG_MEAN = tf.tensor([103.939, 116.779, 123.68]);
    inputs = tf.browser.fromPixels(inputs)
                       .toFloat()
                       .expandDims();
    inputs = tf.reverse(inputs, axis=[-1]);
    inputs = tf.sub(inputs, VGG_MEAN);
    delete VGG_MEAN;

    return inputs;
}


/* Load stylization intensity */
strengthSlider.oninput = (evt) => {
    alpha = tf.tensor2d([[evt.target.value/100.]]);
}


/* Initialize */
$("#model-selector").ready(function()
{
    console.log(tf.getBackend());
    loadModel('Arbitrary');
});
$("#content-image-selector").ready(function()
{
    content.src = 'images/content/avril.jpg';
});
$("#style-image-selector").ready(function()
{
    style.src = 'images/style/udnie.jpg';
});


/* Select model */
$("#model-selector").change(function()
{
    console.log(tf.getBackend());
    loadModel($("#model-selector").val());
});


/* Select content image */
$("#content-image-selector").change(function() 
{
    let name = $("#content-image-selector").val();
    if(name === "file")
    {
        $("#content-file-selector").change(function() 
        {
            let reader = new FileReader();
            reader.onload = function() {
                let dataURL = reader.result;
                $("#content-image").attr("src", dataURL);
            }
            let file = $("#content-file-selector").prop('files')[0];
            reader.readAsDataURL(file);
        });
        $("#content-file-selector").click();
        $("#content-image-selector").val("");
    }
    else
    {
        $("#content-image").attr("src", 'images/content/' + name + '.jpg');
    }
});


/* Select style image */
$("#style-image-selector").change(function() 
{
    let name = $("#style-image-selector").val();
    if(name === "file")
    {
        $("#style-file-selector").change(function() 
        {
            let reader = new FileReader();
            reader.onload = function() {
                let dataURL = reader.result;
                $("#style-image").attr("src", dataURL);
            }
            let file = $("#style-file-selector").prop('files')[0];
            reader.readAsDataURL(file);
        });
        $("#style-file-selector").click();
        $("#style-image-selector").val("");
    }
    else
    {
        $("#style-image").attr("src", 'images/style/' +  name + '.jpg');
    }
});


/* Stylize button */
$("#stylized-button").click(async function()
{
    let content_tensor = await preprocess(content);
    let style_tensor = await preprocess(style);

    const startTime = performance.now();

    if(model === undefined)
    {
        throw new Error("Unknown model name");
    }
    else
    {
        const features = await tf.tidy(() =>{
            return model.predict([content_tensor, style_tensor, alpha]);
        });
        let stylized = features[3];
        // console.log(stylized);
        stylized = stylized.clipByValue(0, 255);

        await tf.browser.toPixels(stylized.toInt().squeeze(), stylizedCanvas);
        
        delete stylized;
        delete features;
    }
    delete content_tensor;
    delete style_tensor;

    /* Message of processing time */
    const totalTime = performance.now() - startTime;
    time(`Taken ${Math.floor(totalTime)} ms`);
});