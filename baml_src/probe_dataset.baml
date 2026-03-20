
class Dataset {
    YesQuestions string[]
    NoQuestions string[]
}

function GenererateProbeDataset(topic: string) -> Dataset {
  // Specify a client as provider/model-name
  // You can also use custom LLM params with a custom client name from clients.baml like "client CustomGPT5" or "client CustomSonnet4"
  client OpenRouterClient
  prompt #"
    You are an expert binary dataset creator. you have been 
    tasked to create 50 question for the given topic where the answer is Yes and an another 50 questions where the answer is No.

    Example Questions with answer Yes for the coding topic:
    - Does Node.js use event loop?
    - Is this the way to print "Hello World". 
      ```
      print("Hello World")
      ```
    - Is SQL prone to hacking?
    - Does golang has reflection?
    - Does VLIW perform better than SIMD?

    Example Questions with answer No for the coding topic:
    - Does RUST has garbage collector?
    - Does set has duplicate entries?
    - is http safer than https?
    - is ssh used to compile C program?
    - is python a type safe language?

    <GUIDELINE>
    - questions must contains enough context when the topic is not common or not well known.
    </GUIDELINE>

    Create question for the topic {{topic}}

    {{ ctx.output_format }}

  "#
}


test dataset_generator {
  functions [GenererateProbeDataset]
  args {
    topic "deceptive manipulation"
  }
}
