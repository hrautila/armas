
BUILT_SOURCES = Doxyfile conf.py

DOXYFLAGS = \
	-Dsrcdir=$(srcdir) \
	-Dbuilddir=$(builddir) \
	-Dtop_builddir=$(top_builddir) \
	-Dtop_srcdir=$(top_srcdir) \
	-Dpackage="\"$(PACKAGE)\"" \
	-Dpackage_name="\"$(PACKAGE_NAME)\"" \
	-Dpackage_version="\"$(PACKAGE_VERSION)\""

SPHINXFLAGS = \
	-Dsrcdir=$(srcdir) \
	-Dbuilddir=$(builddir) \
	-Dtop_builddir=$(top_builddir) \
	-Dtop_srcdir=$(top_srcdir) \
	-Dpackage="\"$(PACKAGE)\"" \
	-Dpackage_name="\"$(PACKAGE_NAME)\"" \
	-Dpackage_version="\"$(PACKAGE_VERSION)\""

Doxyfile: Doxyfile.m4
	m4 $(DOXYFLAGS) $< >$@

conf.py: conf.py.m4
	m4 $(SPHINXFLAGS) $< >$@
