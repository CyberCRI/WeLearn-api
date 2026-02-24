// Defining a data model.
class Resume {
  name string
  email string
  experience string[]
  competencies string[]
}

// Create a function to extract the resume from a string.
function ExtractResume(resume: string) -> Resume {
  client ClientLoadBalancer 
  prompt #"
    Extract from this content:
    {{ resume }}

    {{ ctx.output_format }}
  "#
}



// Test the function with a sample resume. Open the VSCode playground to run this.
test vaibhav_resume {
  functions [ExtractResume]
  args {
    resume #"
      Vaibhav Gupta
      vbv@boundaryml.com

      Experience:
      - Founder at BoundaryML
      - CV Engineer at Google
      - CV Engineer at Microsoft

      Skills:
      - Rust
      - C++
    "#
  }
}
