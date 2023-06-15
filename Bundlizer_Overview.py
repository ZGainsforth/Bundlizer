import streamlit as st

st.title('Bundlizer')
st.markdown('### Produce bundle files from experimental data for submission to SAMIS.')
st.markdown('Questions/Problems/Suggestions: contact Zack Gainsforth zackg@berkeley.edu')

infoStr = 'This tool allows you to upload information from an experiment and produce a bundle file for submission to SAMIS.'+\
'Bundlizer will automatically generate standard data formats and yaml files for your experimental data, '+\
'assuming there is a valid plugin for your instrument and bundle delivery document (BDD).  '+\
'Once you have produced a bundle from your raw data, it can be uploaded directly to SAMIS.  '+\
'Alternately, you can download the raw data plus bundle files for further editing and analysis.  '+\
'The directory containing raw data plus bundle files can be re-uploaded and used to produce a final bundle file.  '+\
'The following flow chart helps explain the flow:'
st.write(infoStr)

st.image('BundlizerFlowChart.png')

### Todos: 
# video on how to use.
# Zip insert files for tiff collection.