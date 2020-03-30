import xml.etree.ElementTree as ET


class XmlParserError(Exception):
    pass

class XmlParser():
    def __init__(self, name):
        try:
            doc = ET.parse(name)
            self.root = doc.getroot()
        except ET.ParseError as pe:
            raise XmlParserError from pe

    def get_config_dict(self):
        try:
            config_dict = {}
            config_dict["ALGORITHM"] = self.root.find('ALGORITHM').text
            config_dict["CXPB"] = float(self.root.find('CXPB').text)
            config_dict["MUTPB"] = float(self.root.find('MUTPB').text)
            config_dict["POP_SIZE"] = int(self.root.find('POP_SIZE').text)
            config_dict["NGEN"] = int(self.root.find('NGEN').text)
            config_dict["MIN"] = float(self.root.find('RANGE').find('MIN').text)
            config_dict["MAX"] = float(self.root.find('RANGE').find('MAX').text)
            mutation = {}
            mutation["name"] = self.root.find('MUTATION').text
            for attr, val in self.root.find('MUTATION').attrib.items():
                mutation[attr] = float(val)
            config_dict["MUTATION"] = mutation
            crossover={}
            crossover["name"] = self.root.find('CROSSOVER').text
            for attr, val in self.root.find('CROSSOVER').attrib.items():
                crossover[attr] = float(val)
            config_dict["CROSSOVER"] = crossover
            return config_dict
        except AttributeError as ae:
            raise XmlParserError from ae