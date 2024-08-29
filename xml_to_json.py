import xmltodict
import json

# Sample ALTO XML string
alto_xml = '''
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
  <Layout>
    <Page WIDTH="2480" HEIGHT="3508">
      <PrintSpace WIDTH="2000" HEIGHT="3000">
        <TextBlock>
          <TextLine>
            <String CONTENT="Hello" WIDTH="500" HEIGHT="50"/>
            <String CONTENT="world!" WIDTH="600" HEIGHT="50"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
'''

# Convert XML to Python dictionary
alto_dict = xmltodict.parse(alto_xml)

# Convert Python dictionary to JSON
alto_json = json.dumps(alto_dict, indent=2)

print(alto_json)
