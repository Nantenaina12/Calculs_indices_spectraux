def classFactory(iface):
    from .ndvi_plugin import NDVICalculatorPlugin
    return NDVICalculatorPlugin(iface)