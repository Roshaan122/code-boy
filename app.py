import streamlit as st
import p2c
import c2p

st.title("Transformer Based Generation")

conversion_type = st.radio("Select conversion type:", ("Pseudo to Code", "Code to Pseudo"))

if conversion_type == "Pseudo to Code":
    pseudo_input = st.text_area("Enter Pseudocode:")
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0)
    if st.button("Generate Code"):
        if pseudo_input.strip():
            with st.spinner("Generating code..."):
                generated_code = p2c.load_model_and_generate("transformer_pseudo2code.pth", pseudo_input, temperature=temperature)
            st.code(generated_code, language="python")
        else:
            st.error("Please enter pseudocode.")
else:
    code_input = st.text_area("Enter Code:")
    if st.button("Generate Pseudocode"):
        if code_input.strip():
            with st.spinner("Generating pseudocode..."):
                generated_pseudo = c2p.generate_output(
                    c2p.model,
                    code_input,
                    c2p.loaded_src_vocab,
                    c2p.loaded_tgt_vocab,
                    c2p.device
                )
            st.text_area("Generated Pseudocode", generated_pseudo)
        else:
            st.error("Please enter code.")