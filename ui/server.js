/**
 * Enrollment Prediction Dashboard - Node.js Server
 * Enterprise UI for Rare Disease Pharmaceutical Enrollment Form Model
 */

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const FLASK_API_URL = process.env.FLASK_API_URL || 'http://localhost:5000';

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// View engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Routes

// Main dashboard
app.get('/', (req, res) => {
    res.render('dashboard', { title: 'Enrollment Prediction Dashboard' });
});

// Prediction page
app.get('/predict', (req, res) => {
    res.render('predict', { title: 'Make Prediction' });
});

// Batch prediction page
app.get('/batch', (req, res) => {
    res.render('batch', { title: 'Batch Prediction' });
});

// Model info page
app.get('/model-info', (req, res) => {
    res.render('model-info', { title: 'Model Information' });
});

// EDA results page
app.get('/eda', (req, res) => {
    res.render('eda', { title: 'Exploratory Data Analysis' });
});

// API proxy routes

// Health check
app.get('/api/health', async (req, res) => {
    try {
        const response = await axios.get(`${FLASK_API_URL}/api/health`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({
            status: 'error',
            message: 'Flask API is not available',
            error: error.message
        });
    }
});

// Single prediction
app.post('/api/predict', async (req, res) => {
    try {
        const response = await axios.post(`${FLASK_API_URL}/api/predict`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({
            status: 'error',
            message: 'Prediction failed',
            error: error.message
        });
    }
});

// Batch prediction
app.post('/api/batch-predict', async (req, res) => {
    try {
        const response = await axios.post(`${FLASK_API_URL}/api/batch_predict`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({
            status: 'error',
            message: 'Batch prediction failed',
            error: error.message
        });
    }
});

// Model info
app.get('/api/model-info', async (req, res) => {
    try {
        const response = await axios.get(`${FLASK_API_URL}/api/model_info`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({
            status: 'error',
            message: 'Failed to get model info',
            error: error.message
        });
    }
});

// EDA results
app.get('/api/eda-results', async (req, res) => {
    try {
        const response = await axios.get(`${FLASK_API_URL}/api/eda_results`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({
            status: 'error',
            message: 'Failed to get EDA results',
            error: error.message
        });
    }
});

// HCP Predictions - main dashboard data
app.get('/api/hcp-predictions', async (req, res) => {
    try {
        const queryString = new URLSearchParams(req.query).toString();
        const response = await axios.get(`${FLASK_API_URL}/api/hcp-predictions?${queryString}`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({
            status: 'error',
            message: 'Failed to get HCP predictions',
            error: error.message
        });
    }
});

// Scores summary
app.get('/api/scores', async (req, res) => {
    try {
        const response = await axios.get(`${FLASK_API_URL}/api/scores`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({
            status: 'error',
            message: 'Failed to get scores',
            error: error.message
        });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Server error:', err);
    res.status(500).json({
        status: 'error',
        message: 'Internal server error',
        error: err.message
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).render('404', { title: 'Page Not Found' });
});

// Start server
app.listen(PORT, () => {
    console.log(`
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   Enrollment Prediction Dashboard                             ║
    ║   Rare Disease Pharmaceutical - Enterprise UI                 ║
    ║                                                               ║
    ║   Server running on: http://localhost:${PORT}                   ║
    ║   Flask API URL: ${FLASK_API_URL}                         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    `);
});

module.exports = app;
